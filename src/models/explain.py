"""
SHAP-based explainability for the fraud model. Answers two different
questions:
  1. Globally, which features drive the model overall?
  2. For a specific flagged transaction, why did it get flagged?
Both matter for a fraud system, a reviewer acting on an alert needs the
per-transaction answer, not just the global one.

Usage:
    python -m src.models.explain
"""
from pathlib import Path
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

from src.models.evaluate import load_validation_set, MODELS_DIR

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models" / "explainability"


def main(sample_size: int = 2000):
    X_val, y_val = load_validation_set()
    model = lgb.Booster(model_file=str(MODELS_DIR / "fraud_model.txt"))

    # SHAP on the full validation set is slow, a random sample is enough
    # to get stable global importance without a multi-hour run
    X_sample = X_val.sample(n=min(sample_size, len(X_val)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # global feature importance
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print(f"Saved global summary plot to {OUTPUT_DIR / 'shap_summary.png'}")

    # bar version, easier to read for a writeup than the dot plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_importance_bar.png", dpi=150)
    plt.close()
    print(f"Saved bar importance plot to {OUTPUT_DIR / 'shap_importance_bar.png'}")

    # per-transaction explanation for one flagged case, picks the
    # highest-scored fraud prediction in the sample as a concrete example
    proba_sample = model.predict(X_sample)
    top_idx = proba_sample.argmax()

    plt.figure()
    shap.force_plot(
        explainer.expected_value, shap_values[top_idx], X_sample.iloc[top_idx],
        matplotlib=True, show=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_single_transaction.png", dpi=150)
    plt.close()
    print(f"Saved single-transaction explanation to {OUTPUT_DIR / 'shap_single_transaction.png'}")
    print(f"(This example transaction scored {proba_sample[top_idx]:.4f} fraud probability)")


if __name__ == "__main__":
    main()