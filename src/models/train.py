"""
Train a LightGBM fraud classifier with a time-based split and MLflow tracking.

Usage:
    python -m src.models.train
"""
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.build_features import build_feature_pipeline

INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def time_based_split(df: pd.DataFrame, val_fraction: float = 0.2):
    """
    Split by TransactionDT, not randomly. A random split lets the model see
    transactions from "the future" relative to what it's being tested on,
    which inflates validation scores in a way that won't hold up on real
    incoming data.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    cutoff = int(len(df) * (1 - val_fraction))
    train_df = df.iloc[:cutoff]
    val_df = df.iloc[cutoff:]
    return train_df, val_df


def get_feature_columns(df: pd.DataFrame) -> list:
    drop_cols = {"TransactionID", "isFraud", "TransactionDT"}
    return [c for c in df.columns if c not in drop_cols]


def train():
    df = pd.read_parquet(INTERIM_DIR / "train_merged.parquet")

    train_df, val_df = time_based_split(df)

    # compute card aggregates on train only, then apply to both splits
    train_df = build_feature_pipeline(train_df, ref_df=train_df)
    val_df = build_feature_pipeline(val_df, ref_df=train_df)

    feature_cols = get_feature_columns(train_df)
    cat_cols = [c for c in feature_cols if train_df[c].dtype == "object"]
    for c in cat_cols:
        train_df[c] = train_df[c].astype("category")
        val_df[c] = val_df[c].astype("category")

    X_train, y_train = train_df[feature_cols], train_df["isFraud"]
    X_val, y_val = val_df[feature_cols], val_df["isFraud"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run():
        params = dict(
            objective="binary",
            metric="auc",
            scale_pos_weight=scale_pos_weight,
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
        )
        mlflow.log_params(params)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
           callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        auc_pr = average_precision_score(y_val, val_proba)
        auc_roc = roc_auc_score(y_val, val_proba)

        mlflow.log_metric("val_auc_pr", auc_pr)
        mlflow.log_metric("val_auc_roc", auc_roc)
        mlflow.lightgbm.log_model(model, "model")

        print(f"Validation AUC-PR: {auc_pr:.4f}")
        print(f"Validation AUC-ROC: {auc_roc:.4f}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(MODELS_DIR / "fraud_model.txt"))
        print(f"Model saved to {MODELS_DIR / 'fraud_model.txt'}")


if __name__ == "__main__":
    train()
