"""
FastAPI endpoint for scoring transactions.

Run locally:
    uvicorn src.api.main:app --reload
"""
from pathlib import Path
from fastapi import FastAPI
import pandas as pd
import lightgbm as lgb

from src.features.build_features import build_feature_pipeline

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
THRESHOLD = 0.22

app = FastAPI(title="Fraud Detection API")
model = None


@app.on_event("startup")
def load_model():
    global model
    model = lgb.Booster(model_file=str(MODELS_DIR / "fraud_model.txt"))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(transaction: dict):
    df = pd.DataFrame([transaction])
    df = build_feature_pipeline(df)  # NOTE: card aggregates need a reference set in production,
                                       # see README for how to handle this at serving time
    proba = float(model.predict(df)[0])
    return {
        "fraud_probability": proba,
        "flagged": proba >= THRESHOLD,
        "threshold_used": THRESHOLD,
    }
