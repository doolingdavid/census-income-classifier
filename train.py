"""LightGBM income classifier for Census Bureau data.

Uses native categorical handling, Optuna hyperparameter optimization with
cross-validation, and a held-out test set for final evaluation.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_PATH = "census-bureau.data"
COLUMNS_PATH = "census-bureau.columns"

NUMERIC_COLS = [
    "age",
    "detailed industry recode",
    "detailed occupation recode",
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "num persons worked for employer",
    "own business or self employed",
    "veterans benefits",
    "weeks worked in year",
]

DROP_COLS = [
    "weight",
    "year",
    "fill inc questionnaire for veteran's admin",
]

RANDOM_STATE = 42


def load_column_names() -> list[str]:
    """Read column names from the .columns file."""
    with open(COLUMNS_PATH) as f:
        names = [line.strip() for line in f if line.strip()]
    # File has 42 feature/label names plus a trailing empty entry
    return names[:42]


def load_data() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load CSV, preprocess columns, return (X, y, sample_weights)."""
    col_names = load_column_names()

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=col_names,
        dtype=str,
        na_filter=False,
    )

    # Strip whitespace from all cells
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Encode label
    y = (df["label"] == "50000+.").astype(int)
    df = df.drop(columns=["label"])

    # Extract sample weights
    sample_weights = pd.to_numeric(df["weight"]).values

    # Drop non-feature columns
    df = df.drop(columns=DROP_COLS)

    # Convert numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col])

    # Convert remaining columns to pd.Categorical (LightGBM auto-detects)
    categorical_cols = [c for c in df.columns if c not in NUMERIC_COLS]
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col])

    print(f"Loaded {len(df):,} rows, {len(df.columns)} features")
    print(f"  Numeric: {len(NUMERIC_COLS)}, Categorical: {len(categorical_cols)}")
    print(f"  Label distribution: 0 (- 50000.) = {(y == 0).sum():,}, "
          f"1 (50000+.) = {(y == 1).sum():,}")
    print(f"  Class ratio: {(y == 0).sum() / (y == 1).sum():.1f}:1")

    return df, y, sample_weights


def split_data(
    X: pd.DataFrame, y: pd.Series, weights: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Stratified 80/20 train/test split."""
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain set: {len(X_train):,} rows")
    print(f"Test set:  {len(X_test):,} rows (held out, used once)")
    return X_train, X_test, y_train, y_test, w_train, w_test


def create_objective(
    X_train: pd.DataFrame, y_train: pd.Series, w_train: np.ndarray
) -> callable:
    """Return an Optuna objective function using 5-fold stratified CV."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "random_state": RANDOM_STATE,
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-5, 100.0, log=True
            ),
            "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 200.0),
            "cat_l2": trial.suggest_float("cat_l2", 1.0, 50.0),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            w_fold_train = w_train[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            w_fold_val = w_train[val_idx]

            model = LGBMClassifier(**params)
            model.fit(
                X_fold_train,
                y_fold_train,
                sample_weight=w_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                eval_sample_weight=[w_fold_val],
                callbacks=[
                    early_stopping(stopping_rounds=50, verbose=False),
                    log_evaluation(period=0),
                ],
                categorical_feature="auto",
            )

            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            fold_auc = roc_auc_score(y_fold_val, y_pred_proba, sample_weight=w_fold_val)
            fold_scores.append(fold_auc)

            # Report intermediate value for pruning
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    return objective


def run_optuna_optimization(
    X_train: pd.DataFrame, y_train: pd.Series, w_train: np.ndarray
) -> dict:
    """Run Optuna study with 50 trials, return best parameters."""
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="lgbm_income_classifier",
    )

    objective = create_objective(X_train, y_train, w_train)
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\n{'=' * 60}")
    print("Optuna Optimization Complete")
    print(f"{'=' * 60}")
    print(f"  Best trial:     #{study.best_trial.number}")
    print(f"  Best CV AUC:    {study.best_value:.6f}")
    print(f"  Completed:      {len(study.trials)} trials")
    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"  Pruned:         {pruned} trials")
    print(f"\n  Best hyperparameters:")
    for key, val in study.best_params.items():
        print(f"    {key}: {val}")

    return study.best_params


def train_final_model(
    best_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: np.ndarray,
) -> LGBMClassifier:
    """Train final model on full training set with early stopping."""
    # 5% internal split for early stopping
    X_tr, X_es, y_tr, y_es, w_tr, w_es = train_test_split(
        X_train, y_train, w_train,
        test_size=0.05, stratify=y_train, random_state=RANDOM_STATE,
    )

    model = LGBMClassifier(
        **best_params,
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=2000,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )

    print(f"\n{'=' * 60}")
    print("Training Final Model")
    print(f"{'=' * 60}")
    print(f"  Training on {len(X_tr):,} rows, early-stop validation on {len(X_es):,} rows")

    model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_es, y_es)],
        eval_sample_weight=[w_es],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=True),
            log_evaluation(period=50),
        ],
        categorical_feature="auto",
    )

    print(f"  Best iteration: {model.best_iteration_}")
    return model


def evaluate_model(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    w_test: np.ndarray,
) -> None:
    """Evaluate model on held-out test set and print metrics."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Core metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba, sample_weight=w_test)
    auc_pr = average_precision_score(y_test, y_pred_proba, sample_weight=w_test)
    f1 = f1_score(y_test, y_pred, sample_weight=w_test)
    precision = precision_score(y_test, y_pred, sample_weight=w_test)
    recall = recall_score(y_test, y_pred, sample_weight=w_test)

    print(f"\n{'=' * 60}")
    print("Hold-Out Test Set Evaluation")
    print(f"{'=' * 60}")
    print(f"  AUC-ROC:    {auc_roc:.6f}")
    print(f"  AUC-PR:     {auc_pr:.6f}")
    print(f"  F1 Score:   {f1:.6f}")
    print(f"  Precision:  {precision:.6f}")
    print(f"  Recall:     {recall:.6f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_w = confusion_matrix(y_test, y_pred, sample_weight=w_test)
    print(f"\n  Confusion Matrix (raw counts):")
    print(f"    {'':>20} Predicted -50K  Predicted 50K+")
    print(f"    {'Actual -50K':>20}  {cm[0, 0]:>10,}    {cm[0, 1]:>10,}")
    print(f"    {'Actual 50K+':>20}  {cm[1, 0]:>10,}    {cm[1, 1]:>10,}")
    print(f"\n  Confusion Matrix (sample-weighted):")
    print(f"    {'':>20} Predicted -50K  Predicted 50K+")
    print(f"    {'Actual -50K':>20}  {cm_w[0, 0]:>14,.0f}    {cm_w[0, 1]:>10,.0f}")
    print(f"    {'Actual 50K+':>20}  {cm_w[1, 0]:>14,.0f}    {cm_w[1, 1]:>10,.0f}")

    # Classification report
    print(f"\n  Classification Report (weighted):")
    report = classification_report(
        y_test, y_pred,
        target_names=["- 50000.", "50000+."],
        sample_weight=w_test,
    )
    for line in report.split("\n"):
        print(f"    {line}")

    # Feature importances (top 15)
    importances = model.feature_importances_
    feature_names = model.feature_name_
    indices = np.argsort(importances)[::-1][:15]
    print(f"\n  Top 15 Feature Importances (split-based):")
    for rank, idx in enumerate(indices, 1):
        print(f"    {rank:>2}. {feature_names[idx]:<45} {importances[idx]:>6}")


def save_artifacts(
    model: LGBMClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    w_train: np.ndarray,
    w_test: np.ndarray,
    best_params: dict,
) -> None:
    """Save model, data splits, and feature importances to artifacts/."""
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Save model
    joblib.dump(model, artifacts_dir / "model.pkl")

    # Save train/test data splits
    joblib.dump({
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "w_train": w_train, "w_test": w_test,
    }, artifacts_dir / "data_splits.pkl")

    # Save feature importances
    importances = pd.DataFrame({
        "feature": model.feature_name_,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(artifacts_dir / "feature_importances.csv", index=False)

    # Save best params as JSON for easy re-use
    serializable_params = {k: float(v) if isinstance(v, (np.floating,)) else v
                           for k, v in best_params.items()}
    with open(artifacts_dir / "best_params.json", "w") as f:
        json.dump(serializable_params, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Artifacts Saved")
    print(f"{'=' * 60}")
    print(f"  artifacts/model.pkl")
    print(f"  artifacts/data_splits.pkl")
    print(f"  artifacts/feature_importances.csv")
    print(f"  artifacts/best_params.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM income classifier")
    parser.add_argument(
        "--skip-optuna", action="store_true",
        help="Skip Optuna search and use saved best params from artifacts/best_params.json",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LightGBM Income Classifier — Census Bureau Data")
    print("=" * 60)

    X, y, weights = load_data()
    X_train, X_test, y_train, y_test, w_train, w_test = split_data(X, y, weights)

    if args.skip_optuna:
        params_path = Path("artifacts/best_params.json")
        if not params_path.exists():
            print("ERROR: --skip-optuna requires artifacts/best_params.json from a prior run")
            return
        with open(params_path) as f:
            best_params = json.load(f)
        print(f"\nSkipping Optuna — loaded params from {params_path}")
    else:
        best_params = run_optuna_optimization(X_train, y_train, w_train)

    model = train_final_model(best_params, X_train, y_train, w_train)
    evaluate_model(model, X_test, y_test, w_test)
    save_artifacts(model, X_train, X_test, y_train, y_test, w_train, w_test, best_params)


if __name__ == "__main__":
    main()
