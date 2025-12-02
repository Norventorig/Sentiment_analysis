from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from model.core import model_handler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


app = FastAPI(title="Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "frontend"), name="static")


class PredictRequest(BaseModel):
    text: str


@app.post("/api/predict")
async def predict(req: PredictRequest):
    label, confidence = model_handler.predict(req.text)
    return {"success": True,
            "result": {"label": label, "confidence": float(confidence)}}


@app.get("/")
async def index():
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")
