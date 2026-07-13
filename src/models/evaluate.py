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

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# adjust these to whatever your writeup argues for
COST_FALSE_NEGATIVE = 500  # missed fraud
COST_FALSE_POSITIVE = 5    # blocked legitimate transaction, customer friction


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


def main(X_val, y_val):
    model = lgb.Booster(model_file=str(MODELS_DIR / "fraud_model.txt"))
    y_proba = model.predict(X_val)

    auc_pr = average_precision_score(y_val, y_proba)
    print(f"AUC-PR: {auc_pr:.4f}")

    results = evaluate_at_thresholds(y_val, y_proba)
    print(results.to_string(index=False))

    best_threshold = find_best_threshold(y_val, y_proba)
    print(f"\nCost-minimizing threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    # load your validation set here and call main(X_val, y_val)
    print("Import and call main(X_val, y_val) with your validation data.")
