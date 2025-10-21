"""Threshold optimisation utilities."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from .metrics import compute_metrics


def search_threshold(grid: Iterable[float], y_true: np.ndarray, y_prob: np.ndarray, optimize_for: str) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {}
    best_score = -np.inf
    for threshold in grid:
        metrics = compute_metrics(y_true, y_prob, threshold=threshold)
        score = metrics.get(optimize_for, -np.inf)
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


__all__ = ["search_threshold"]
