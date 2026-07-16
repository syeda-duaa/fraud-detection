"""
Unit tests for the threshold/cost logic in src/models/evaluate.py.

These use small synthetic arrays, not the real dataset or a trained model,
so they run in under a second and don't depend on data/raw/ being populated.
That's on purpose: this is testing the *logic*, not the model.
"""
import numpy as np
import pandas as pd

from src.models.evaluate import evaluate_at_thresholds, find_best_threshold


def make_synthetic_labels():
    """
    10 examples, 3 are actual fraud (label 1). Scores are hand-picked so we
    know exactly what should happen at different thresholds.
    """
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.55, 0.80, 0.90])
    return y_true, y_proba


def test_evaluate_at_thresholds_returns_expected_columns():
    y_true, y_proba = make_synthetic_labels()
    results = evaluate_at_thresholds(y_true, y_proba)

    expected_cols = {"threshold", "precision", "recall",
                      "false_positives", "false_negatives", "total_cost"}
    assert expected_cols.issubset(set(results.columns))
    assert len(results) > 0


def test_perfect_separation_gives_perfect_recall_at_low_threshold():
    """If every fraud case scores higher than every non-fraud case, a low
    enough threshold should catch all fraud with zero false negatives."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    results = evaluate_at_thresholds(y_true, y_proba, thresholds=[0.5])
    row = results.iloc[0]

    assert row["recall"] == 1.0
    assert row["false_negatives"] == 0


def test_high_threshold_reduces_recall():
    """Recall should never increase as the threshold goes up, this is a
    monotonicity sanity check on the underlying confusion-matrix logic."""
    y_true, y_proba = make_synthetic_labels()
    results = evaluate_at_thresholds(y_true, y_proba, thresholds=[0.1, 0.5, 0.9])

    recalls = results.sort_values("threshold")["recall"].tolist()
    assert recalls[0] >= recalls[1] >= recalls[2]


def test_find_best_threshold_returns_value_in_valid_range():
    y_true, y_proba = make_synthetic_labels()
    best = find_best_threshold(y_true, y_proba)

    assert 0.0 <= best <= 1.0


def test_higher_false_negative_cost_favors_lower_threshold():
    """
    If missing fraud is made extremely expensive relative to false alarms,
    the cost-minimizing threshold should not increase, since a higher
    threshold only ever costs more false negatives, never fewer.
    """
    y_true, y_proba = make_synthetic_labels()

    results = evaluate_at_thresholds(y_true, y_proba, thresholds=np.arange(0.05, 0.95, 0.05))

    # cheap false positives, very expensive false negatives
    cheap_fp_cost = results["false_negatives"] * 10000 + results["false_positives"] * 1
    expensive_fp_cost = results["false_negatives"] * 500 + results["false_positives"] * 500

    threshold_cheap_fp = results.loc[cheap_fp_cost.idxmin(), "threshold"]
    threshold_balanced = results.loc[expensive_fp_cost.idxmin(), "threshold"]

    assert threshold_cheap_fp <= threshold_balanced