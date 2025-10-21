"""Data loading and filtering utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..utils.config import DataConfig


def load_diabetes_data(config: DataConfig) -> pd.DataFrame:
    """Load the diabetes dataset applying all configured filters."""
    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    id_cols = config.identifier_cols

    # normalize column names (some csvs store uppercase headers)
    df.columns = [c.strip() for c in df.columns]

    filters = config.filters
    if "time_in_hospital" in df.columns:
        df = df[(df["time_in_hospital"] >= filters.min_los) & (df["time_in_hospital"] <= filters.max_los)]

    discharge_col = config.columns.discharge_disposition_col
    if discharge_col in df.columns:
        df = df[~df[discharge_col].isin(filters.exclude_discharge_to_ids)]

    if filters.first_encounter_per_patient:
        patient_col = id_cols.patient_id
        encounter_col = id_cols.encounter_id
        df = df.sort_values(by=[patient_col, encounter_col])
        df = df.drop_duplicates(subset=patient_col, keep="first")

    target_col = config.target.name
    positive_values = set(config.target.positive_values)
    df["target"] = df[target_col].isin(positive_values).astype(int)
    return df.reset_index(drop=True)


__all__ = ["load_diabetes_data"]
