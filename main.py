
import os
import io
import json
import uuid
import sqlite3
import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
import fitz
from openai import OpenAI
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=API_KEY or "missing",
    base_url=os.getenv("OPENAI_BASE_URL") or None
)

db = sqlite3.connect(
    str(BASE_DIR / "interviews.db"),
    check_same_thread=False
)
db.execute("CREATE TABLE IF NOT EXISTS s(id TEXT PRIMARY KEY, data TEXT)")
db.commit()

LIM = {"intro": 20, "skills": 40, "projects": 40}
MAX_MB = 5

app = FastAPI(title="AI Voice Interviewer")


def save(sid, data):
    db.execute(
        "INSERT OR REPLACE INTO s VALUES(?, ?)",
        (sid, json.dumps(data))
    )
    db.commit()


def load(sid):
    row = db.execute(
        "SELECT data FROM s WHERE id=?", (sid,)
    ).fetchone()

    if not row:
        raise HTTPException(404, "Session not found")

    return json.loads(row[0])


def llm(system, user):
    if not API_KEY:
        raise HTTPException(
            503,
            "OPENAI_API_KEY is not configured in Render environment variables."
        )

    error = None

    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )
            return json.loads(response.choices[0].message.content)

        except Exception as exc:
            error = exc

    raise HTTPException(502, f"LLM request failed: {error}")


PLAN_SYS = """You are an interview designer. Use ONLY facts in the resume;
never invent skills, projects or experience.
Return JSON: {"candidate_name":str,
"intro":{"question":str,"rubric":[str]},
"skills":[{"question":str,"rubric":[str]}],
"projects":[{"question":str,"rubric":[str]}]}
Ask a professional introduction question, 5 questions about listed skills,
and up to 4 questions about listed projects.
Use only information supported by the resume.
Rubrics should contain 3-6 short key points."""


@app.post("/api/resume")
async def resume(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    raw = await file.read()

    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_MB} MB).")

    try:
        document = fitz.open(stream=raw, filetype="pdf")
        text = "\n".join(page.get_text() for page in document)
        document.close()
    except Exception:
        raise HTTPException(400, "This is not a valid PDF.")

    if len(text.strip()) < 200:
        raise HTTPException(
            400,
            "No readable text found. Please upload a text-based PDF."
        )

    plan = llm(PLAN_SYS, text[:12000])

    try:
        questions = []

        def add(section, items, total):
            items = [
                item for item in items
                if item.get("question")
            ][:5]

            if not items:
                raise ValueError(section)

            for item in items:
                questions.append({
                    "id": len(questions),
                    "section": section,
                    "question": item["question"],
                    "rubric": item.get("rubric", []),
                    "max": round(total / len(items), 2)
                })

        add("intro", [plan["intro"]], 20)
        add("skills", plan["skills"], 40)
        add("projects", plan["projects"], 40)

    except Exception:
        raise HTTPException(
            502,
            "The AI returned an invalid interview plan. Please try again."
        )

    sid = uuid.uuid4().hex[:12]

    save(sid, {
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": plan.get("candidate_name", ""),
        "resume": text[:12000],
        "questions": questions,
        "answers": {}
    })

    return {
        "session_id": sid,
        "candidate": plan.get("candidate_name", ""),
        "questions": [
            {
                key: question[key]
                for key in ("id", "section", "question", "max")
            }
            for question in questions
        ]
    }


EVAL_SYS = """Grade the interview answer by meaning and technical correctness,
not exact wording. Accept valid paraphrases and synonyms.
Give partial credit. Deduct for technical errors and irrelevant answers.
Return JSON: {"marks":number,"reason":str,"correct":[str],
"missing":[str],"improved_answer":str}"""


class Ans(BaseModel):
    session_id: str
    question_id: int
    transcript: str


@app.post("/api/answer")
def answer(a: Ans):
    data = load(a.session_id)

    if a.question_id < 0 or a.question_id >= len(data["questions"]):
        raise HTTPException(400, "Invalid question.")

    if a.question_id > len(data["answers"]):
        raise HTTPException(409, "Answer earlier questions first.")

    if len(a.transcript.strip()) < 3:
        raise HTTPException(400, "Please record or type your answer.")

    question = data["questions"][a.question_id]

    evaluation = llm(
        EVAL_SYS,
        json.dumps({
            "question": question["question"],
            "max_marks": question["max"],
            "rubric": question["rubric"],
            "resume_excerpt": data["resume"][:4000],
            "answer": a.transcript[:4000]
        })
    )

    try:
        marks = round(
            min(max(float(evaluation["marks"]), 0), question["max"]), 1
        )
    except Exception:
        raise HTTPException(502, "The AI returned an invalid score.")

    data["answers"][str(question["id"])] = {
        "transcript": a.transcript,
        "marks": marks,
        "reason": str(evaluation.get("reason", "")),
        "correct": evaluation.get("correct", []),
        "missing": evaluation.get("missing", []),
        "improved": str(evaluation.get("improved_answer", ""))
    }

    data.pop("summary", None)
    save(a.session_id, data)

    result = data["answers"][str(question["id"])]

    return {
        "marks": marks,
        "max": question["max"],
        "reason": result["reason"],
        "correct": result["correct"],
        "missing": result["missing"],
        "improved": result["improved"]
    }


def totals(data):
    sections = {key: 0 for key in LIM}

    for question in data["questions"]:
        answer_data = data["answers"].get(str(question["id"]))
        if answer_data:
            sections[question["section"]] += answer_data["marks"]

    sections = {
        key: round(min(max(value, 0), LIM[key]), 1)
        for key, value in sections.items()
    }

    return sections, round(sum(sections.values()), 1)


SUM_SYS = """Given graded interview results, return JSON:
{"strengths":[str],"weaknesses":[str],"suggestions":[str],
"preparation_plan":[str]}.
Be specific to the actual answers."""


def final(data):
    if len(data["answers"]) < len(data["questions"]):
        raise HTTPException(409, "Interview not finished.")

    sections, total = totals(data)

    if "summary" not in data:
        digest = [
            {
                "q": question["question"],
                "section": question["section"],
                "marks": data["answers"][str(question["id"])]["marks"],
                "max": question["max"],
                "missing": data["answers"][str(question["id"])]["missing"]
            }
            for question in data["questions"]
        ]
        data["summary"] = llm(SUM_SYS, json.dumps(digest))

    return {
        "sections": sections,
        "limits": LIM,
        "total": total,
        **data["summary"]
    }


@app.get("/api/result/{sid}")
def result(sid: str):
    data = load(sid)
    output = final(data)
    save(sid, data)
    return output


@app.get("/api/report/{sid}")
def report(sid: str):
    data = load(sid)
    output = final(data)
    save(sid, data)

    styles = getSampleStyleSheet()
    story = []
    buffer = io.BytesIO()

    def paragraph(value, style="BodyText"):
        story.append(
            Paragraph(
                escape(str(value)).replace("\n", "<br/>"),
                styles[style]
            )
        )

    def bullet_list(items):
        for item in items:
            paragraph("• " + str(item))

    paragraph("AI Voice Interviewer: Interview Report", "Title")
    paragraph(
        f"Date: {data['created']} | Session: {sid} | "
        f"Candidate: {data['name'] or 'N/A'}"
    )

    table = Table([
        ["Section", "Score"],
        ["Professional Introduction", f"{output['sections']['intro']} / 20"],
        ["Technical Skills", f"{output['sections']['skills']} / 40"],
        ["Projects", f"{output['sections']['projects']} / 40"],
        ["TOTAL", f"{output['total']} / 100"]
    ])

    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 4), (-1, 4), "Helvetica-Bold")
    ]))

    story.extend([Spacer(1, 10), table, Spacer(1, 10)])

    for heading, key in [
        ("Strengths", "strengths"),
        ("Areas for Improvement", "weaknesses"),
        ("Suggestions", "suggestions"),
        ("Personalized Preparation Plan", "preparation_plan")
    ]:
        paragraph(heading, "Heading2")
        bullet_list(output.get(key, []))

    paragraph("Question-by-Question Results", "Heading2")

    for index, question in enumerate(data["questions"], 1):
        answer_data = data["answers"][str(question["id"])]

        paragraph(f"Q{index} [{question['section']}] {question['question']}", "Heading4")
        paragraph(f"Marks: {answer_data['marks']} / {question['max']}")
        paragraph("Your answer: " + answer_data["transcript"])
        paragraph("Why: " + answer_data["reason"])
        paragraph("Correct concepts:")
        bullet_list(answer_data["correct"] or ["None"])
        paragraph("Missing / incorrect:")
        bullet_list(answer_data["missing"] or ["None"])
        paragraph("Suggested improved answer: " + answer_data["improved"])
        story.append(Spacer(1, 8))

    SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title="AI Voice Interviewer Report"
    ).build(story)

    return Response(
        buffer.getvalue(),
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f"attachment; filename=interview_report_{sid}.pdf"
        }
    )


@app.delete("/api/session/{sid}")
def delete(sid: str):
    db.execute("DELETE FROM s WHERE id=?", (sid,))
    db.commit()
    return {"deleted": True}


@app.get("/")
def home():
    html_path = BASE_DIR / "index.html"

    if not html_path.is_file():
        raise HTTPException(
            500,
            "index.html not found. Keep index.html beside main.py."
        )

    return FileResponse(str(html_path), media_type="text/html")
