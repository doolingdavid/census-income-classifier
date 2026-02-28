"""Global SHAP feature importance plot from the saved LightGBM model.

Loads artifacts and produces a SHAP beeswarm summary plot to validate
the feature importance ordering used in the parallel coordinates HiPlot.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    print("Loading artifacts...")
    model = joblib.load(ARTIFACTS_DIR / "model.pkl")
    data = joblib.load(ARTIFACTS_DIR / "data_splits.pkl")
    X_test = data["X_test"]

    # Subsample for speed — 5,000 rows is plenty for stable SHAP estimates
    rng = np.random.RandomState(42)
    n = min(5000, len(X_test))
    idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test.iloc[idx]

    print(f"Computing SHAP values on {n:,} test samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values is a list of two arrays.
    # Use class 1 (50K+) SHAP values.
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Beeswarm summary plot
    print("Generating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(sv, X_sample, show=False, max_display=20)
    plt.tight_layout()
    out_path = ARTIFACTS_DIR / "shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # Also print the mean |SHAP| ranking for easy comparison
    mean_abs_shap = np.abs(sv).mean(axis=0)
    feature_names = X_sample.columns.tolist()
    ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])

    print(f"\n{'=' * 60}")
    print("Global SHAP Feature Importance (mean |SHAP value|)")
    print(f"{'=' * 60}")
    for rank, (name, val) in enumerate(ranking, 1):
        print(f"  {rank:>2}. {name:<45} {val:.4f}")


if __name__ == "__main__":
    main()
