import os, io, json, uuid, sqlite3, datetime
from xml.sax.saxutils import escape
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import fitz
from openai import OpenAI
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "missing"), base_url=os.getenv("OPENAI_BASE_URL") or None)
db = sqlite3.connect(os.path.join(os.path.dirname(__file__), "interviews.db"), check_same_thread=False)
db.execute("create table if not exists s(id text primary key, data text)")
LIM = {"intro": 20, "skills": 40, "projects": 40}
MAX_MB = 5
app = FastAPI(title="AI Voice Interviewer")

def save(sid, d):
    db.execute("insert or replace into s values(?,?)", (sid, json.dumps(d))); db.commit()

def load(sid):
    r = db.execute("select data from s where id=?", (sid,)).fetchone()
    if not r: raise HTTPException(404, "Session not found")
    return json.loads(r[0])

def llm(system, user):
    err = None
    for _ in range(2):  # one retry for transient errors / bad JSON
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0.2,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            err = e
    raise HTTPException(502, f"LLM request failed: {err}")

PLAN_SYS = """You are an interview designer. Use ONLY facts in the resume; never invent skills, projects or experience.
Return JSON: {"candidate_name":str,"intro":{"question":str,"rubric":[str]},
"skills":[{"question":str,"rubric":[str]}],"projects":[{"question":str,"rubric":[str]}]}
- intro: ask for a professional introduction (education, skills, project experience). Rubric = resume facts a good answer may mention.
- skills: 5 questions on technologies actually listed (what/why/how/use case/difference). Moderate difficulty, no repeats. Rubric = key concepts a correct answer contains.
- projects: up to 4 questions on projects actually listed (problem, objective, tech, model choice, preprocessing, challenges, evaluation, improvements). Spread across projects. Only ask what the resume supports. If no projects exist, ask 2 questions about the most substantial work/experience listed.
- If a section has little material, ask fewer (minimum 2) questions. Rubrics: 3-6 short key points each."""

@app.post("/api/resume")
async def resume(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    raw = await file.read()
    if len(raw) > MAX_MB * 1024 * 1024: raise HTTPException(400, f"File too large (max {MAX_MB} MB).")
    try:
        text = "\n".join(p.get_text() for p in fitz.open(stream=raw, filetype="pdf"))
    except Exception:
        raise HTTPException(400, "This is not a valid PDF.")
    if len(text.strip()) < 200:
        raise HTTPException(400, "No readable text found. This looks like a scanned PDF; OCR would be required. Please upload a text-based PDF.")
    plan = llm(PLAN_SYS, text[:12000])
    try:
        qs = []
        def add(sec, items, total):
            items = [i for i in items if i.get("question")][:5]
            if not items: raise ValueError(sec)
            for i in items:
                qs.append({"id": len(qs), "section": sec, "question": i["question"],
                           "rubric": i.get("rubric", []), "max": round(total / len(items), 2)})
        add("intro", [plan["intro"]], 20); add("skills", plan["skills"], 40); add("projects", plan["projects"], 40)
    except Exception:
        raise HTTPException(502, "The AI returned an invalid interview plan. Please try again.")
    sid = uuid.uuid4().hex[:12]
    save(sid, {"created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "name": plan.get("candidate_name", ""),
               "resume": text[:12000], "questions": qs, "answers": {}})
    return {"session_id": sid, "candidate": plan.get("candidate_name", ""),
            "questions": [{k: q[k] for k in ("id", "section", "question", "max")} for q in qs]}

EVAL_SYS = """You grade a spoken interview answer (transcribed by speech-to-text) by MEANING and technical correctness, never exact wording.
Steps: identify the concept tested; use the rubric key points; accept paraphrases, synonyms and valid alternative explanations; give partial credit;
deduct for technical errors/contradictions/irrelevance; never reward confident but wrong answers or keyword stuffing; ignore grammar, accent and obvious transcription slips.
Scoring: marks = max * (0.6 correctness of concepts + 0.2 completeness + 0.2 clarity/depth/practical example), each 0..1. Empty or irrelevant = 0.
For the intro question also judge relevance, professionalism and consistency with the resume (contradicting the resume lowers the score).
Return JSON: {"marks":number,"reason":str,"correct":[str],"missing":[str],"improved_answer":str}"""

class Ans(BaseModel):
    session_id: str
    question_id: int
    transcript: str

@app.post("/api/answer")
def answer(a: Ans):
    d = load(a.session_id)
    if a.question_id < 0 or a.question_id >= len(d["questions"]): raise HTTPException(400, "Invalid question.")
    if a.question_id > len(d["answers"]): raise HTTPException(409, "Answer earlier questions first.")
    if len(a.transcript.strip()) < 3: raise HTTPException(400, "Transcript is empty. Please record or type your answer.")
    q = d["questions"][a.question_id]
    ev = llm(EVAL_SYS, json.dumps({"question": q["question"], "max_marks": q["max"], "rubric": q["rubric"],
                                   "resume_excerpt": d["resume"][:4000], "answer": a.transcript[:4000]}))
    try:
        marks = round(min(max(float(ev["marks"]), 0), q["max"]), 1)
    except Exception:
        raise HTTPException(502, "The AI returned an invalid score. Please resubmit.")
    d["answers"][str(q["id"])] = {"transcript": a.transcript, "marks": marks, "reason": str(ev.get("reason", "")),
        "correct": ev.get("correct", []), "missing": ev.get("missing", []), "improved": str(ev.get("improved_answer", ""))}
    d.pop("summary", None); save(a.session_id, d)
    return {"marks": marks, "max": q["max"], **{k: d["answers"][str(q["id"])][k] for k in ("reason", "correct", "missing", "improved")}}

def totals(d):
    sec = {k: 0 for k in LIM}
    for q in d["questions"]:
        a = d["answers"].get(str(q["id"]))
        if a: sec[q["section"]] += a["marks"]
    sec = {k: round(min(max(v, 0), LIM[k]), 1) for k, v in sec.items()}
    return sec, round(min(max(sum(sec.values()), 0), 100), 1)

SUM_SYS = """Given graded interview results, return JSON: {"strengths":[str],"weaknesses":[str],"suggestions":[str],"preparation_plan":[str]}.
Be specific to the actual answers; 3-5 items each; preparation_plan is a short ordered study plan."""

def final(d):
    if len(d["answers"]) < len(d["questions"]): raise HTTPException(409, "Interview not finished.")
    sec, total = totals(d)
    if "summary" not in d:
        digest = [{"q": q["question"], "section": q["section"], "marks": d["answers"][str(q["id"])]["marks"], "max": q["max"],
                   "missing": d["answers"][str(q["id"])]["missing"]} for q in d["questions"]]
        d["summary"] = llm(SUM_SYS, json.dumps(digest))
    return {"sections": sec, "limits": LIM, "total": total, **d["summary"]}

@app.get("/api/result/{sid}")
def result(sid: str):
    d = load(sid); r = final(d); save(sid, d); return r

@app.get("/api/report/{sid}")
def report(sid: str):
    d = load(sid); r = final(d); save(sid, d)
    st = getSampleStyleSheet(); S = []; buf = io.BytesIO()
    def P(t, s="BodyText"): S.append(Paragraph(escape(str(t)).replace("\n", "<br/>"), st[s]))
    def L(items): [P("• " + x) for x in items]
    P("AI Voice Interviewer: Interview Report", "Title")
    P(f"Date: {d['created']}   |   Session: {sid}   |   Candidate: {d['name'] or 'N/A'}")
    t = Table([["Section", "Score"], ["Professional Introduction", f"{r['sections']['intro']} / 20"],
               ["Technical Skills", f"{r['sections']['skills']} / 40"], ["Projects", f"{r['sections']['projects']} / 40"],
               ["TOTAL", f"{r['total']} / 100"]])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                           ("FONTNAME", (0, 4), (-1, 4), "Helvetica-Bold")]))
    S += [Spacer(1, 10), t, Spacer(1, 10)]
    for h, k in (("Strengths", "strengths"), ("Areas for Improvement", "weaknesses"), ("Suggestions", "suggestions"),
                 ("Personalized Preparation Plan", "preparation_plan")):
        P(h, "Heading2"); L(r.get(k, []))
    P("Question-by-Question Results", "Heading2")
    for i, q in enumerate(d["questions"], 1):
        a = d["answers"][str(q["id"])]
        P(f"Q{i} [{q['section']}] {q['question']}", "Heading4")
        P(f"Marks: {a['marks']} / {q['max']}"); P("Your answer: " + a["transcript"]); P("Why: " + a["reason"])
        P("Correct concepts:"); L(a["correct"] or ["None"]); P("Missing / incorrect:"); L(a["missing"] or ["None"])
        P("Suggested improved answer: " + a["improved"]); S.append(Spacer(1, 8))
    P("How This Application Works", "Heading2")
    P("1) You upload a text-based PDF; PyMuPDF extracts the text. 2) An LLM builds personalized questions and rubrics from the resume only. "
      "3) The browser speaks each question (Web Speech synthesis) and transcribes your voice (Web Speech recognition); you can edit the transcript. "
      "4) The backend sends question, rubric, resume excerpt and transcript to the LLM, which grades by meaning and technical correctness. "
      "5) The backend clamps scores (intro 20, skills 40, projects 40, total 100), stores them in SQLite, and ReportLab generates this PDF.")
    SimpleDocTemplate(buf, pagesize=A4, title="AI Voice Interviewer Report").build(S)
    return Response(buf.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=interview_report_{sid}.pdf"})

@app.delete("/api/session/{sid}")
def delete(sid: str):
    db.execute("delete from s where id=?", (sid,)); db.commit(); return {"deleted": True}

@app.get("/")
def home():
    return {
        "message": "AI Voice Interviewer API is running",
        "docs": "/docs"
    }
