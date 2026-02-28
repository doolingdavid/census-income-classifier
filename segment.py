"""Segmentation analysis: parallel coordinates + interpretable decision tree.

Loads the trained LightGBM model and data splits from artifacts/,
then produces:
  - Parallel coordinate plots (matplotlib + HiPlot interactive HTML)
  - A shallow decision tree for human-readable segmentation rules
"""

from pathlib import Path

import hiplot as hip
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ARTIFACTS_DIR = Path("artifacts")
LABEL_NAMES = {0: "- 50000.", 1: "50000+."}


def load_artifacts():
    """Load saved model, data splits, and feature importances."""
    model = joblib.load(ARTIFACTS_DIR / "model.pkl")
    data = joblib.load(ARTIFACTS_DIR / "data_splits.pkl")
    importances = pd.read_csv(ARTIFACTS_DIR / "feature_importances.csv")

    # LightGBM converts spaces to underscores in feature_name_.
    # Map the underscore names back to the actual DataFrame column names.
    actual_cols = data["X_train"].columns.tolist()
    lgbm_to_actual = {}
    for col in actual_cols:
        lgbm_name = col.replace(" ", "_")
        lgbm_to_actual[lgbm_name] = col
    importances["feature"] = importances["feature"].map(
        lambda f: lgbm_to_actual.get(f, f)
    )

    return model, data, importances


def compute_shap_importances(model, X: pd.DataFrame) -> pd.DataFrame:
    """Compute global SHAP feature importance (mean |SHAP|) on a subsample."""
    rng = np.random.RandomState(42)
    n = min(5000, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    X_sample = X.iloc[idx]

    print("  Computing SHAP values on {:,} samples...".format(n))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Binary classifier: use class 1 (50K+) SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    importances = pd.DataFrame({
        "feature": X_sample.columns.tolist(),
        "importance": mean_abs,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return importances


def _encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns to integers for plotting/tree input."""
    X_enc = X.copy()
    for col in X_enc.columns:
        if hasattr(X_enc[col], "cat"):
            X_enc[col] = X_enc[col].cat.codes.astype(float)
            # Replace -1 (NaN sentinel) with NaN
            X_enc[col] = X_enc[col].replace(-1, np.nan)
    return X_enc


def make_parallel_coordinates_matplotlib(
    X: pd.DataFrame, y: pd.Series, importances: pd.DataFrame, top_n: int = 15
) -> None:
    """Create a static parallel coordinates plot colored by income label."""
    top_features = importances["feature"].head(top_n).tolist()
    X_sub = X[top_features].copy()
    X_enc = _encode_categoricals(X_sub)

    # Normalize to [0, 1] for visual comparability
    scaler = MinMaxScaler()
    X_norm = pd.DataFrame(
        scaler.fit_transform(X_enc.fillna(0)),
        columns=top_features,
        index=X_enc.index,
    )
    X_norm["label"] = y.values

    # Subsample: 3,000 from each class for readability
    samples = []
    rng = np.random.RandomState(42)
    for label in [0, 1]:
        mask = X_norm["label"] == label
        pool = X_norm[mask]
        n = min(3000, len(pool))
        samples.append(pool.sample(n=n, random_state=rng))
    df_plot = pd.concat(samples)

    # Draw majority class (0) first, minority (1) on top
    df_maj = df_plot[df_plot["label"] == 0]
    df_min = df_plot[df_plot["label"] == 1]

    fig, ax = plt.subplots(figsize=(20, 8))
    x_ticks = range(top_n)

    for _, row in df_maj.iterrows():
        ax.plot(x_ticks, row[top_features].values, color="steelblue", alpha=0.06, linewidth=0.5)
    for _, row in df_min.iterrows():
        ax.plot(x_ticks, row[top_features].values, color="orangered", alpha=0.08, linewidth=0.5)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Normalized feature value (0–1)")
    ax.set_title(f"Parallel Coordinates — Top {top_n} Features by Importance")

    # Legend
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color="steelblue", linewidth=2, label="- 50000."),
        Line2D([0], [0], color="orangered", linewidth=2, label="50000+."),
    ]
    ax.legend(handles=legend_lines, loc="upper right", fontsize=11)

    plt.tight_layout()
    out_path = ARTIFACTS_DIR / "parallel_coordinates.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_hiplot_html(
    X: pd.DataFrame, y: pd.Series, importances: pd.DataFrame, top_n: int = 15
) -> None:
    """Create an interactive HiPlot parallel coordinates HTML file."""
    top_features = importances["feature"].head(top_n).tolist()
    X_sub = X[top_features].copy()
    X_enc = _encode_categoricals(X_sub)
    X_enc["label"] = y.values.astype(str)
    X_enc["label"] = X_enc["label"].map({"0": "- 50000.", "1": "50000+."})

    # Subsample: 5,000 per class for performance
    samples = []
    rng = np.random.RandomState(42)
    for label in ["- 50000.", "50000+."]:
        mask = X_enc["label"] == label
        pool = X_enc[mask]
        n = min(5000, len(pool))
        samples.append(pool.sample(n=n, random_state=rng))
    df_sample = pd.concat(samples)

    # Build list of dicts for HiPlot, with columns in SHAP importance order
    # Use OrderedDict so HiPlot preserves most-important-first ordering
    from collections import OrderedDict
    col_order = ["label"] + top_features
    data = [OrderedDict((k, row[k]) for k in col_order) for _, row in df_sample.iterrows()]
    experiment = hip.Experiment.from_iterable(data)

    # Explicitly set the parallel plot column order: label first, then by SHAP importance
    experiment.display_data(hip.Displays.PARALLEL_PLOT).update({
        "order": col_order,
    })

    out_path = ARTIFACTS_DIR / "parallel_coordinates.html"
    experiment.to_html(str(out_path))
    print(f"  Saved: {out_path}")


def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    w: np.ndarray,
    importances: pd.DataFrame,
    top_n: int = 10,
) -> tuple[DecisionTreeClassifier, list[str]]:
    """Train a shallow decision tree on the top-N most important features."""
    top_features = importances["feature"].head(top_n).tolist()
    X_sub = _encode_categoricals(X[top_features].copy()).fillna(0)

    tree = DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X_sub, y, sample_weight=w)

    # Print training metrics
    y_pred = tree.predict(X_sub)
    acc = accuracy_score(y, y_pred, sample_weight=w)
    f1 = f1_score(y, y_pred, sample_weight=w)
    print(f"\n  Decision Tree (train) — Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return tree, top_features


def evaluate_tree(
    tree: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    w_test: np.ndarray,
    feature_names: list[str],
) -> None:
    """Evaluate the decision tree on the held-out test set."""
    X_sub = _encode_categoricals(X_test[feature_names].copy()).fillna(0)
    y_pred = tree.predict(X_sub)

    acc = accuracy_score(y_test, y_pred, sample_weight=w_test)
    f1 = f1_score(y_test, y_pred, sample_weight=w_test)
    print(f"  Decision Tree (test)  — Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print()
    print("  Classification Report (test set):")
    report = classification_report(
        y_test, y_pred,
        target_names=["- 50000.", "50000+."],
        sample_weight=w_test,
    )
    for line in report.split("\n"):
        print(f"    {line}")


def print_tree_rules(tree: DecisionTreeClassifier, feature_names: list[str]) -> None:
    """Print human-readable rules, save to text file, and save visual tree diagram."""
    print("\n" + "=" * 60)
    print("Decision Tree Segmentation Rules")
    print("=" * 60)
    rules = export_text(tree, feature_names=feature_names, show_weights=True)
    print(rules)

    # Save text rules to file
    rules_path = ARTIFACTS_DIR / "decision_tree_rules.txt"
    sep = "=" * 60
    features_str = ", ".join(feature_names)
    lines = [
        "Decision Tree Segmentation Rules",
        sep,
        f"Features used ({len(feature_names)}): {features_str}",
        f"Max depth: {tree.get_depth()}, Leaves: {tree.get_n_leaves()}",
        sep,
        "",
        rules,
    ]
    with open(rules_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {rules_path}")

    # Visual tree diagram
    fig, ax = plt.subplots(figsize=(28, 12))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["- 50000.", "50000+."],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    plt.tight_layout()
    out_path = ARTIFACTS_DIR / "decision_tree.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    print("=" * 60)
    print("Segmentation Analysis")
    print("=" * 60)

    model, data, _split_importances = load_artifacts()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    w_train, w_test = data["w_train"], data["w_test"]

    print(f"\nLoaded artifacts — {len(X_train):,} train / {len(X_test):,} test rows")

    # Compute SHAP-based feature importance (replaces split-based ordering)
    print(f"\n{'=' * 60}")
    print("SHAP Feature Importance")
    print(f"{'=' * 60}")
    importances = compute_shap_importances(model, X_test)
    print(f"\n  Top 15 features (SHAP):")
    for i, row in importances.head(15).iterrows():
        print(f"    {i + 1:>2}. {row['feature']:<45} {row['importance']:.4f}")

    # Save SHAP importances for reference
    importances.to_csv(ARTIFACTS_DIR / "shap_importances.csv", index=False)

    # Parallel coordinates on test set (ordered by SHAP importance)
    print(f"\n{'=' * 60}")
    print("Parallel Coordinates (test set, SHAP ordering)")
    print(f"{'=' * 60}")
    make_parallel_coordinates_matplotlib(X_test, y_test, importances, top_n=15)
    make_hiplot_html(X_test, y_test, importances, top_n=15)

    # Decision tree on training data (top SHAP features)
    print(f"\n{'=' * 60}")
    print("Decision Tree Segmentation (top 10 SHAP features)")
    print(f"{'=' * 60}")
    tree, feature_names = train_decision_tree(
        X_train, y_train, w_train, importances, top_n=10
    )
    evaluate_tree(tree, X_test, y_test, w_test, feature_names)
    print_tree_rules(tree, feature_names)


if __name__ == "__main__":
    main()
