"""Model evaluation: metrics, diagnostic plots, and permutation importance."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend so training works headless.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import ALL_FEATURES, RANDOM_STATE


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute classification metrics for a set of predictions."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_plots(y_true, y_pred, y_proba, reports_dir: Path) -> None:
    """Save confusion matrix and ROC curve figures to `reports_dir`."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Not placed", "Placed"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Final model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(reports_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def permutation_importance_report(pipeline, X_test: pd.DataFrame, y_test, reports_dir: Path) -> pd.DataFrame:
    """Measure each raw feature's contribution to ROC-AUC via permutation."""
    result = permutation_importance(
        pipeline,
        X_test[ALL_FEATURES],
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    reports_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(reports_dir / "permutation_importance.csv", index=False)
    return importance
