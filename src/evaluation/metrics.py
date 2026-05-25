from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error


def brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray, labels: List[int]) -> float:
    y_true = np.asarray(y_true)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y_idx = np.vectorize(label_to_index.get)(y_true)
    y_onehot = np.eye(len(labels))[y_idx]
    return float(np.mean(np.sum((y_onehot - y_proba) ** 2, axis=1)))


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: List[int],
) -> Dict[str, float]:
    y_pred = np.array(labels)[np.argmax(y_proba, axis=1)]

    metrics = {
        "log_loss": float(log_loss(y_true, y_proba, labels=labels)),
        "brier_score": brier_score_multiclass(y_true, y_proba, labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    return metrics
