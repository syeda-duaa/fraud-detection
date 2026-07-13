"""
Evaluate a trained model with metrics that actually matter for imbalanced fraud data,
plus a cost-based threshold sweep.

Usage:
    python -m src.models.evaluate
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

from src.features.build_features import build_feature_pipeline
from src.models.train import time_based_split, get_feature_columns

INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# adjust these to whatever your writeup argues for
COST_FALSE_NEGATIVE = 200  # missed fraud
COST_FALSE_POSITIVE = 20    # blocked legitimate transaction, customer friction


def evaluate_at_thresholds(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
        rows.append(dict(threshold=t, precision=precision, recall=recall,
                          false_positives=fp, false_negatives=fn, total_cost=cost))

    return pd.DataFrame(rows)


def find_best_threshold(y_true, y_proba) -> float:
    """Pick the threshold that minimizes total business cost, not the default 0.5."""
    results = evaluate_at_thresholds(y_true, y_proba, thresholds=np.arange(0.01, 0.99, 0.01))
    best_row = results.loc[results["total_cost"].idxmin()]
    return best_row["threshold"]


def load_validation_set():
    """
    Rebuilds the exact same validation split and features used in training.
    Kept identical to train.py on purpose, mismatched preprocessing between
    training and evaluation is its own quiet leakage/bug source.
    """
    df = pd.read_parquet(INTERIM_DIR / "train_merged.parquet")
    train_df, val_df = time_based_split(df)

    train_df = build_feature_pipeline(train_df, ref_df=train_df)
    val_df = build_feature_pipeline(val_df, ref_df=train_df)

    feature_cols = get_feature_columns(train_df)
    cat_cols = [c for c in feature_cols if val_df[c].dtype == "object"]
    for c in cat_cols:
        val_df[c] = val_df[c].astype("category")

    return val_df[feature_cols], val_df["isFraud"]


def main(X_val=None, y_val=None):
    if X_val is None or y_val is None:
        X_val, y_val = load_validation_set()

    model = lgb.Booster(model_file=str(MODELS_DIR / "fraud_model.txt"))
    y_proba = model.predict(X_val)

    auc_pr = average_precision_score(y_val, y_proba)
    print(f"AUC-PR: {auc_pr:.4f}")

    results = evaluate_at_thresholds(y_val, y_proba)
    print(results.to_string(index=False))

    best_threshold = find_best_threshold(y_val, y_proba)
    print(f"\nCost-minimizing threshold: {best_threshold:.2f}")

    return results, best_threshold


if __name__ == "__main__":
    main()