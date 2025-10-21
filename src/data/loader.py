"""Data loading and filtering utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from ..utils.config import DataConfig


def load_diabetes_data(config: DataConfig) -> pd.DataFrame:
    """Load the diabetes dataset applying all configured filters."""
    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    _ensure_columns_present(df, _required_columns(config))

    filters = config.filters
    los_col = "time_in_hospital"
    if los_col in df.columns:
        df = df[(df[los_col] >= filters.min_los) & (df[los_col] <= filters.max_los)]

    discharge_col = config.columns.discharge_disposition_col
    df = df[~df[discharge_col].isin(filters.exclude_discharge_to_ids)]

    if filters.first_encounter_per_patient:
        patient_col = config.identifier_cols.patient_id
        encounter_col = config.identifier_cols.encounter_id
        df = (
            df.sort_values([patient_col, encounter_col])
            .drop_duplicates(subset=patient_col, keep="first")
            .copy()
        )

    target_col = config.target.name
    positive_values = {str(v) for v in config.target.positive_values}
    df["target"] = df[target_col].astype(str).isin(positive_values).astype(int)
    return df.reset_index(drop=True)


def _required_columns(config: DataConfig) -> List[str]:
    cols: List[str] = [
        config.identifier_cols.encounter_id,
        config.identifier_cols.patient_id,
        config.target.name,
        config.columns.discharge_disposition_col,
    ]
    cols.extend(config.columns.numeric)
    cols.extend(config.columns.categorical_low_card)
    cols.extend(config.columns.icd_cols)
    cols.extend(config.columns.drug_cols)
    cols.append(config.columns.hospital_col)
    cols.append(config.columns.specialty_col)
    cols.append(config.columns.admission_source_col)
    cols.append(config.columns.admission_type_col)
    return list(dict.fromkeys(cols))


def _ensure_columns_present(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Dataset missing required columns: {missing}")


__all__ = ["load_diabetes_data"]
