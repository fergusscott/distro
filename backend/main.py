import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from parser import parse_submission
from scorer import score_all_carriers, get_agency_stats, train_all, ALL_FEATURES
from carriers import CARRIERS

app = FastAPI(title="Distro — Submission Intelligence for Brokers")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


class SubmissionText(BaseModel):
    text: str


@app.on_event("startup")
async def startup():
    train_all()


@app.get("/api/carriers")
async def carriers():
    return [
        {"id": c["id"], "name": c["name"], "tagline": c["tagline"]}
        for c in CARRIERS.values()
    ]


@app.post("/api/analyze")
async def analyze(body: SubmissionText):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Submission text is required")
    try:
        parsed = parse_submission(body.text)
        scores = score_all_carriers(parsed)
        return {"parsed": parsed, "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/network")
async def network():
    return get_agency_stats()


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
