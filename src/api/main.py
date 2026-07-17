"""
FastAPI endpoint for scoring transactions.

Run locally:
    uvicorn src.api.main:app --reload
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI
import lightgbm as lgb

from src.features.build_features import build_feature_pipeline_for_serving, load_card_reference

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CARD_REFERENCE_PATH = MODELS_DIR / "card_reference.parquet"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
THRESHOLD = 0.22

app = FastAPI(title="Fraud Detection API")
model = None
card_stats = None
feature_cols = None
categorical_cols = None


@app.on_event("startup")
def load_model():
    global model, card_stats, feature_cols, categorical_cols
    model = lgb.Booster(model_file=str(MODELS_DIR / "fraud_model.txt"))
    card_stats = load_card_reference(CARD_REFERENCE_PATH)
    with open(FEATURE_SCHEMA_PATH) as f:
        schema = json.load(f)
    feature_cols = schema["feature_cols"]
    categorical_cols = schema["categorical_cols"]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "card_reference_loaded": card_stats is not None,
        "schema_loaded": feature_cols is not None,
    }


def build_model_input(transaction: dict) -> pd.DataFrame:
    raw_df = pd.DataFrame([transaction])
    engineered = build_feature_pipeline_for_serving(raw_df, card_stats)

    row = {col: np.nan for col in feature_cols}
    for col in feature_cols:
        if col in engineered.columns:
            row[col] = engineered[col].iloc[0]

    df = pd.DataFrame([row], columns=feature_cols)
    for col, categories in categorical_cols.items():
        df[col] = pd.Categorical(df[col], categories=categories)

    return df


@app.post("/predict")
def predict(transaction: dict):
    df = build_model_input(transaction)
    proba = float(model.predict(df)[0])
    return {
        "fraud_probability": proba,
        "flagged": proba >= THRESHOLD,
        "threshold_used": THRESHOLD,
    }