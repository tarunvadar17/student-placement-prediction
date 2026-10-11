"""Training pipeline: model comparison, tuning, final fit, and persistence.

Usage:
    python -m placement.train [--data PATH] [--samples N] [--no-tune]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .config import (
    ALL_FEATURES,
    CV_FOLDS,
    DECISION_THRESHOLD,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET,
    TEST_SIZE,
)
from .data_generator import ensure_dataset
from .evaluate import compute_metrics, permutation_importance_report, save_plots
from .preprocessing import build_preprocessor, load_dataset, validate_schema

logger = logging.getLogger(__name__)

MODEL_FILENAME = "placement_model.joblib"


def get_candidate_models() -> dict:
    """Return unfitted candidate estimators keyed by name."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


PARAM_GRIDS = {
    "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
    "random_forest": {
        "model__max_depth": [None, 8, 12],
        "model__min_samples_leaf": [3, 5, 10],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
    },
}


def build_pipeline(estimator) -> Pipeline:
    """Combine preprocessing and an estimator into one pipeline."""
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])


def compare_models(X: pd.DataFrame, y: pd.Series) -> tuple[dict, str]:
    """Cross-validate each candidate and return the summary and best model name."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    summary: dict[str, dict] = {}

    for name, estimator in get_candidate_models().items():
        scores = cross_validate(
            build_pipeline(estimator),
            X,
            y,
            cv=cv,
            scoring=["roc_auc", "f1", "accuracy"],
            n_jobs=-1,
        )
        summary[name] = {
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
            "f1_mean": float(scores["test_f1"].mean()),
            "accuracy_mean": float(scores["test_accuracy"].mean()),
        }
        logger.info(
            "%-20s ROC-AUC %.4f ± %.4f | F1 %.4f",
            name,
            summary[name]["roc_auc_mean"],
            summary[name]["roc_auc_std"],
            summary[name]["f1_mean"],
        )

    best = max(summary, key=lambda k: summary[k]["roc_auc_mean"])
    logger.info("Selected model: %s", best)
    return summary, best


def fit_final_model(X_train, y_train, model_name: str, tune: bool) -> tuple[Pipeline, dict]:
    """Fit the selected model, optionally tuning hyperparameters with grid search."""
    pipeline = build_pipeline(get_candidate_models()[model_name])
    best_params: dict = {}

    if tune and model_name in PARAM_GRIDS:
        search = GridSearchCV(
            pipeline,
            PARAM_GRIDS[model_name],
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        best_params = search.best_params_
        logger.info("Best hyperparameters: %s", best_params)
    else:
        pipeline.fit(X_train, y_train)

    return pipeline, best_params


def run_training(
    df: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    tune: bool = True,
) -> dict:
    """Run the full training workflow and return a summary dictionary."""
    validate_schema(df, require_target=True)
    X = df[ALL_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train size: %d | Test size: %d | Positive rate: %.3f", len(X_train), len(X_test), y.mean())

    cv_summary, best_name = compare_models(X_train, y_train)
    pipeline, best_params = fit_final_model(X_train, y_train, best_name, tune)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    test_metrics = compute_metrics(y_test, y_pred, y_proba)
    logger.info("Held-out test ROC-AUC: %.4f | F1: %.4f", test_metrics["roc_auc"], test_metrics["f1"])

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    save_plots(y_test, y_pred, y_proba, reports_dir)
    permutation_importance_report(pipeline, X_test, y_test, reports_dir)

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "model_name": best_name,
        "best_params": best_params,
        "features": ALL_FEATURES,
        "decision_threshold": DECISION_THRESHOLD,
        "trained_at": trained_at,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, models_dir / MODEL_FILENAME)

    summary = {
        "best_model": best_name,
        "best_params": best_params,
        "cv": cv_summary,
        "test": test_metrics,
        "trained_at": trained_at,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info("Model saved to %s", models_dir / MODEL_FILENAME)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the student placement model.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="Path to training CSV.")
    parser.add_argument("--samples", type=int, default=5000, help="Synthetic rows if data is absent.")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    data_path = ensure_dataset(args.data, n_samples=args.samples)
    df = load_dataset(data_path)
    summary = run_training(df, tune=not args.no_tune)
    print(json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
