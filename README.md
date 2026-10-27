# AI Voice Interviewer

An AI-powered mock interview application. Upload your resume as a PDF, answer personalized interview questions by voice, and get scored feedback with a downloadable PDF report.



---

## Features

- **Resume-based questions:** Extracts text from a text-based PDF resume using PyMuPDF and uses an LLM to generate questions only from facts in the resume.
- **Three interview sections:** Professional Introduction (20 marks), Technical Skills (40 marks), and Projects (40 marks), for a total of 100.
- **Voice interaction:** The browser reads each question aloud (Web Speech synthesis) and transcribes your spoken answer (Web Speech recognition). You can edit the transcript before submitting.
- **Meaning-based grading:** The LLM grades each answer against a rubric by meaning and technical correctness, not exact wording. It gives partial credit and deducts for technical errors.
- **Detailed feedback:** For every answer you get marks, a reason, correct concepts, missing points, and a suggested improved answer.
- **Final summary:** Strengths, areas for improvement, suggestions, and a personalized preparation plan.
- **PDF report:** Download the complete interview report generated with ReportLab.
- **Privacy control:** Delete a session and its stored data at any time.

## How It Works

1. You upload a text-based PDF resume. PyMuPDF extracts the text.
2. An LLM builds an interview plan (questions and rubrics) using only the resume content.
3. The browser speaks each question and transcribes your voice answer.
4. The backend sends the question, rubric, resume excerpt, and your transcript to the LLM, which returns a score and feedback in JSON.
5. The backend clamps scores to each section's limit (intro 20, skills 40, projects 40, total 100) and saves them in SQLite.
6. After the last question, the LLM writes a summary and ReportLab generates the PDF report.

## Tech Stack

| Area | Technologies |
|---|---|
| Language | Python, JavaScript, HTML |
| Backend | FastAPI, Uvicorn, Pydantic |
| AI / LLM | OpenAI API (default model `gpt-4o-mini`), any OpenAI-compatible provider, prompt engineering, structured JSON output |
| Resume parsing | PyMuPDF |
| Database | SQLite |
| Reports | ReportLab |
| Voice | Web Speech API (speech recognition and speech synthesis) |
| Config | python-dotenv |

## Project Structure

```
ai-voice-interviewer/
├── backend/
│   └── main.py          # FastAPI app, LLM prompts, scoring, PDF report
├── frontend/
│   └── index.html       # Single-page UI with voice input and output
├── .env.example         # Environment variable template
└── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- An API key for an OpenAI-compatible LLM provider
- Google Chrome or Microsoft Edge (recommended for the Web Speech API)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/babu-001/[REPO-NAME].git
cd [REPO-NAME]

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your environment file
cp .env.example .env            # On Windows: copy .env.example .env
```

### Configuration

Edit `.env`:

```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=
LLM_MODEL=gpt-4o-mini
```

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | API key for your LLM provider |
| `OPENAI_BASE_URL` | Leave empty for OpenAI. For Groq use `https://api.groq.com/openai/v1`. Gemini's OpenAI endpoint and Ollama also work. |
| `LLM_MODEL` | Model name, default `gpt-4o-mini` |

> Never commit your `.env` file. Only `.env.example` should be in the repository.

### Run the App

From the project root:

```bash
uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000 in Chrome or Edge.

## Usage

1. Upload your resume (PDF, text-based, maximum 5 MB).
2. Allow microphone access when the browser asks.
3. Listen to each question, then record your answer or type it.
4. Review and edit the transcript, then submit.
5. Read the feedback for each answer, then continue to the next question.
6. After the last question, view your scores and download the PDF report.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/resume` | Upload a resume PDF and create an interview session |
| POST | `/api/answer` | Submit a transcript for a question and get graded feedback |
| GET | `/api/result/{sid}` | Get section scores, total, and the AI summary |
| GET | `/api/report/{sid}` | Download the interview report as a PDF |
| DELETE | `/api/session/{sid}` | Delete a session |

## Design Notes

- **Prompt-engineered pipeline:** Separate system prompts handle interview planning, answer grading, and final summary.
- **Structured output:** Requests use JSON mode, with one retry if the model call fails or returns invalid JSON.
- **Grounded questions:** The planning prompt instructs the model to use only facts in the resume and never invent skills or projects.
- **Score safety:** Marks are clamped to each question's maximum and each section's limit, so the model cannot return out-of-range scores.
- **Ordered flow:** Questions must be answered in order.

## Limitations

- Only text-based PDFs are supported. Scanned resumes would need OCR, which is not included.
- Only the first 12,000 characters of the resume are used.
- Speech recognition depends on the browser. Chrome and Edge work best, and support in other browsers varies.
- LLM grading can vary slightly between runs. Scores are a practice aid, not an official assessment.
- Sessions are stored in a local SQLite file (`interviews.db`), which is not suitable for multi-server deployment as is.

## Future Improvements

- OCR support for scanned resumes
- Interview history and progress tracking across sessions
- Role-specific question sets (for example Data Scientist or AI Engineer)
- Docker support and cloud deployment guide




