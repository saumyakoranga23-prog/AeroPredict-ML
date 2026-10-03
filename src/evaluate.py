from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.preprocessing import FAILURE_LABELS


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R²": float(r2_score(actual, predicted)),
    }


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "Precision": float(precision_score(actual, predicted, labels=FAILURE_LABELS, average="weighted", zero_division=0)),
        "Recall": float(recall_score(actual, predicted, labels=FAILURE_LABELS, average="weighted", zero_division=0)),
        "F1-score": float(f1_score(actual, predicted, labels=FAILURE_LABELS, average="weighted", zero_division=0)),
        "Confusion Matrix": confusion_matrix(actual, predicted, labels=FAILURE_LABELS).tolist(),
    }
