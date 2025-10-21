"""Calibration utilities for Platt scaling and isotonic regression."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationResult:
    method: str
    model: object

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            probs = 1.0 / (1.0 + np.exp(-logits))
            return self.model.transform(probs)
        if self.method == "platt":
            logits_2d = logits.reshape(-1, 1)
            return self.model.predict_proba(logits_2d)[:, 1]
        raise ValueError(f"Unsupported calibration method {self.method}")


def fit_calibrator(method: str, logits: np.ndarray, labels: np.ndarray) -> CalibrationResult:
    if method == "isotonic":
        reg = IsotonicRegression(out_of_bounds="clip")
        probs = 1.0 / (1.0 + np.exp(-logits))
        reg.fit(probs, labels)
        return CalibrationResult(method="isotonic", model=reg)
    if method == "platt":
        clf = LogisticRegression(solver="lbfgs")
        clf.fit(logits.reshape(-1, 1), labels)
        return CalibrationResult(method="platt", model=clf)
    raise ValueError(f"Unsupported calibration method {method}")


__all__ = ["CalibrationResult", "fit_calibrator"]
