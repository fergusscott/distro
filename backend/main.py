import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any
from dotenv import load_dotenv

load_dotenv(override=False)  # don't override env vars already set by the host

from parser import parse_submission, parse_pdf
from scorer import score_all_carriers, get_agency_stats, train_all
from carriers import CARRIERS
from config import get_config, save_config, reset_config
from book import get_book
from goals import get_goals, save_goals, generate_recommendations

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


class ConfigBody(BaseModel):
    config: dict[str, Any]


class GoalsBody(BaseModel):
    goals: dict[str, Any]


@app.on_event("startup")
async def startup():
    train_all()


@app.get("/api/carriers")
async def carriers():
    return [{"id": c["id"], "name": c["name"], "tagline": c["tagline"]} for c in CARRIERS.values()]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        pdf_bytes = await file.read()
        parsed = parse_pdf(pdf_bytes)
        scores = score_all_carriers(parsed)
        return {"parsed": parsed, "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/api/carriers/{carrier_id}/config")
async def get_carrier_config(carrier_id: str):
    if carrier_id not in CARRIERS:
        raise HTTPException(status_code=404, detail="Carrier not found")
    return get_config(carrier_id)


@app.post("/api/carriers/{carrier_id}/config")
async def update_carrier_config(carrier_id: str, body: ConfigBody):
    if carrier_id not in CARRIERS:
        raise HTTPException(status_code=404, detail="Carrier not found")
    return save_config(carrier_id, body.config)


@app.post("/api/carriers/{carrier_id}/config/reset")
async def reset_carrier_config(carrier_id: str):
    if carrier_id not in CARRIERS:
        raise HTTPException(status_code=404, detail="Carrier not found")
    return reset_config(carrier_id)


@app.get("/api/carriers/{carrier_id}/book")
async def carrier_book(carrier_id: str):
    if carrier_id not in CARRIERS:
        raise HTTPException(status_code=404, detail="Carrier not found")
    return get_book(carrier_id)


@app.get("/api/carriers/{carrier_id}/goals")
async def carrier_goals(carrier_id: str):
    if carrier_id not in CARRIERS:
        raise HTTPException(status_code=404, detail="Carrier not found")
    return get_goals(carrier_id)


@app.post("/api/carriers/{carrier_id}/goals")
async def update_carrier_goals(carrier_id: str, body: GoalsBody):
    if carrier_id not in CARRIERS:
        raise HTTPException(status_code=404, detail="Carrier not found")
    return save_goals(carrier_id, body.goals)


@app.get("/api/carriers/{carrier_id}/recommendations")
async def carrier_recommendations(carrier_id: str):
    if carrier_id not in CARRIERS:
        raise HTTPException(status_code=404, detail="Carrier not found")
    book = get_book(carrier_id)
    goals = get_goals(carrier_id)
    recs = generate_recommendations(carrier_id, book, goals)
    return {"recommendations": recs}


@app.get("/api/network")
async def network():
    return get_agency_stats()


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
