"""Preprocessing utilities responsible for imputing and scaling data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..utils.config import ColumnsConfig, PreprocessingConfig


@dataclass
class PreprocessArtifacts:
    numeric_imputer: SimpleImputer
    categorical_imputer: SimpleImputer
    scaler: object
    categorical_levels: Dict[str, List[str]]


class TabularPreprocessor:
    """Fit imputers/scalers on training split and transform other splits."""

    def __init__(self, config: PreprocessingConfig, columns: ColumnsConfig) -> None:
        self.config = config
        self.columns = columns
        self.numeric_imputer = SimpleImputer(strategy=config.numeric_imputer)
        self.categorical_imputer = SimpleImputer(strategy=config.categorical_imputer)
        scaler_cls = StandardScaler if config.scaler == "standard" else MinMaxScaler
        self.scaler = scaler_cls()
        self.categorical_levels: Dict[str, List[str]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        numeric = df[self.columns.numeric]
        categoricals = df[self.columns.categorical_low_card]
        self.numeric_imputer.fit(numeric)
        if not categoricals.empty:
            self.categorical_imputer.fit(categoricals)
            for col in categoricals.columns:
                levels = pd.Series(categoricals[col]).fillna("__NA__").astype(str)
                counts = levels.value_counts()
                kept = counts[counts >= self.config.min_freq_for_category].index.tolist()
                if self.config.use_unknown_category and self.config.unknown_label not in kept:
                    kept.append(self.config.unknown_label)
                self.categorical_levels[col] = kept
        imputed_numeric = self.numeric_imputer.transform(numeric)
        self.scaler.fit(imputed_numeric)

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        numeric = df[self.columns.numeric]
        categoricals = df[self.columns.categorical_low_card]
        numeric_arr = self.numeric_imputer.transform(numeric)
        numeric_scaled = self.scaler.transform(numeric_arr)

        cat_arrays: Dict[str, np.ndarray] = {}
        if not categoricals.empty:
            filled = self.categorical_imputer.transform(categoricals)
            for i, col in enumerate(categoricals.columns):
                levels = self.categorical_levels.get(col, [])
                mapping = {value: idx for idx, value in enumerate(levels)}
                column_values = []
                for raw in filled[:, i]:
                    key = str(raw)
                    if key not in mapping:
                        key = self.config.unknown_label if self.config.use_unknown_category else levels[0]
                    column_values.append(mapping[key])
                cat_arrays[col] = np.asarray(column_values, dtype=np.int64)
        return numeric_scaled, categoricals.values if not categoricals.empty else np.empty((len(df), 0)), cat_arrays

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        self.fit(df)
        return self.transform(df)

    def to_artifacts(self) -> PreprocessArtifacts:
        return PreprocessArtifacts(
            numeric_imputer=self.numeric_imputer,
            categorical_imputer=self.categorical_imputer,
            scaler=self.scaler,
            categorical_levels=self.categorical_levels,
        )


__all__ = ["TabularPreprocessor", "PreprocessArtifacts"]
