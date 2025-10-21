"""Load and validate IDS mapping tables used for hierarchy construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ADMISSION_TYPE_MAP = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "NULL",
    7: "Trauma Center",
    8: "Not Mapped",
}

DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to Medicare approved swing bed",
    16: "Discharged/transferred/referred another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "NULL",
    19: "Expired at home. Medicaid only, hospice.",
    20: "Expired in a medical facility. Medicaid only, hospice.",
    21: "Expired, place unknown. Medicaid only, hospice.",
    22: "Discharged/transferred to another rehab fac including rehab units of a hospital .",
    23: "Discharged/transferred to a long term care hospital.",
    24: "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Discharged/transferred to a federal health care facility.",
    28: "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
    29: "Discharged/transferred to a Critical Access Hospital (CAH).",
    30: "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
}

ADMISSION_SOURCE_MAP = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from a hospital",
    5: "Transfer from a Skilled Nursing Facility (SNF)",
    6: "Transfer from another health care facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from critial access hospital",
    11: "Normal Delivery",
    12: "Premature Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    15: "Not Available",
    17: "NULL",
    18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency",
    20: "Not Mapped",
    21: "Unknown/Invalid",
    22: "Transfer from hospital inpt/same fac reslt in a sep claim",
    23: "Born inside this hospital",
    24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center",
    26: "Transfer from Hospice",
}


@dataclass
class HierarchyMappings:
    icd_to_group: Dict[str, str]
    drug_to_class: Dict[str, str]
    admission_type_labels: Dict[int, str]
    discharge_disposition_labels: Dict[int, str]
    admission_source_labels: Dict[int, str]


DEFAULT_MAPPINGS = HierarchyMappings(
    icd_to_group={},
    drug_to_class={},
    admission_type_labels=ADMISSION_TYPE_MAP,
    discharge_disposition_labels=DISCHARGE_DISPOSITION_MAP,
    admission_source_labels=ADMISSION_SOURCE_MAP,
)


def _load_mapping_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found at {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Only CSV mapping files are supported")


def load_mappings(mapping_path: Optional[str]) -> HierarchyMappings:
    if mapping_path is None:
        return DEFAULT_MAPPINGS

    df = _load_mapping_file(Path(mapping_path))
    columns = {col.lower(): col for col in df.columns}

    icd_to_group: Dict[str, str] = {}
    if {"icd_code", "icd_group"}.issubset(columns):
        icd_to_group = dict(zip(df[columns["icd_code"]], df[columns["icd_group"]]))

    drug_to_class: Dict[str, str] = {}
    if {"drug", "drug_class"}.issubset(columns):
        drug_to_class = dict(zip(df[columns["drug"]], df[columns["drug_class"]]))

    return HierarchyMappings(
        icd_to_group=icd_to_group,
        drug_to_class=drug_to_class,
        admission_type_labels=ADMISSION_TYPE_MAP,
        discharge_disposition_labels=DISCHARGE_DISPOSITION_MAP,
        admission_source_labels=ADMISSION_SOURCE_MAP,
    )


__all__ = ["HierarchyMappings", "load_mappings", "DEFAULT_MAPPINGS"]
