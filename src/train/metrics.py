"""Metric computation utilities for evaluation and logging."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn import metrics


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if mask.any():
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            ece += abs(avg_pred - avg_true) * mask.mean()
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        "auroc": metrics.roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "auprc": metrics.average_precision_score(y_true, y_prob),
        "brier": metrics.brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "precision_pos": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall_pos": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1_pos": metrics.f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": metrics.balanced_accuracy_score(y_true, y_pred),
    }
    return results


__all__ = ["compute_metrics", "expected_calibration_error"]
