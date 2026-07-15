"""
Cost sensitivity analysis: shows how the optimal decision threshold shifts
under different assumptions about the relative cost of a missed fraud
(false negative) vs. a wrongly blocked legitimate transaction (false positive).

This is the piece that turns "here's a threshold" into "here's why that
threshold, and here's how it moves if the business's risk tolerance is
different." Run this once, drop the resulting table into your writeup.

Usage:
    python -m src.models.cost_sensitivity
"""
import pandas as pd
import lightgbm as lgb

from src.models.evaluate import load_validation_set, evaluate_at_thresholds, MODELS_DIR

# each tuple is (false_negative_cost, false_positive_cost, label)
SCENARIOS = [
    (500, 5, "Baseline: fraud is costly, friction is cheap (100:1)"),
    (500, 20, "Higher friction cost: customers annoyed by false alarms (25:1)"),
    (200, 20, "Lower-value fraud, e.g. smaller average transaction size (10:1)"),
    (1000, 5, "High-value fraud, e.g. enterprise/wire transactions (200:1)"),
    (500, 50, "Strict customer-experience business (10:1)"),
]


def run_sensitivity():
    X_val, y_val = load_validation_set()
    model = lgb.Booster(model_file=str(MODELS_DIR / "fraud_model.txt"))
    y_proba = model.predict(X_val)  # score once, reuse across all scenarios

    rows = []
    for fn_cost, fp_cost, label in SCENARIOS:
        results = evaluate_at_thresholds(y_val, y_proba)
        # recompute cost under this scenario's assumptions rather than the
        # hardcoded ones baked into evaluate_at_thresholds's default
        results["total_cost"] = (
            results["false_negatives"] * fn_cost + results["false_positives"] * fp_cost
        )
        best_row = results.loc[results["total_cost"].idxmin()]
        rows.append(dict(
            scenario=label,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
            cost_ratio=f"{fn_cost // fp_cost}:1",
            best_threshold=round(best_row["threshold"], 2),
            precision=round(best_row["precision"], 3),
            recall=round(best_row["recall"], 3),
            false_positives=int(best_row["false_positives"]),
            false_negatives=int(best_row["false_negatives"]),
        ))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_sensitivity()
    pd.set_option("display.max_colwidth", 60)
    print(df.to_string(index=False))
    df.to_csv("cost_sensitivity_results.csv", index=False)
    print("\nSaved to cost_sensitivity_results.csv")