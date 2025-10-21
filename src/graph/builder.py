"""Build :class:`torch_geometric.data.HeteroData` objects from tabular data."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ..data.mappings import HierarchyMappings
from ..data.vocab import Vocab
from ..utils.config import ColumnsConfig, GraphConfig, PreprocessingConfig

DRUG_STATUS_VALUES = {"Up", "Down", "Steady", "No"}
STATUS_TO_INDEX = {"Up": 0, "Down": 1, "Steady": 2, "No": 3}


class GraphBuilder:
    """Construct heterogeneous graphs following the project specification."""

    def __init__(
        self,
        graph_config: GraphConfig,
        preprocessing: PreprocessingConfig,
        columns: ColumnsConfig,
        mappings: HierarchyMappings,
        vocabs: Dict[str, Vocab],
    ) -> None:
        self.graph_config = graph_config
        self.preprocessing = preprocessing
        self.columns = columns
        self.mappings = mappings
        self.vocabs = vocabs

    def build(
        self,
        df: pd.DataFrame,
        encounter_features: np.ndarray,
        encounter_cat_indices: Dict[str, np.ndarray],
    ) -> HeteroData:
        df_local = df.reset_index(drop=True)
        num_rows = len(df_local)
        if encounter_features.shape[0] != num_rows:
            raise ValueError("Encounter feature matrix row count does not match dataframe")

        data = HeteroData()
        data["encounter"].num_nodes = num_rows
        data["encounter"].x = torch.as_tensor(encounter_features, dtype=torch.float32)
        data["encounter"].y = torch.as_tensor(df_local["target"].values, dtype=torch.float32)

        self._add_encounter_categorical_features(data, encounter_cat_indices, num_rows)
        self._add_entity_nodes(data)
        self._add_edges(data, df_local)
        return data

    def _add_encounter_categorical_features(
        self,
        data: HeteroData,
        encounter_cat_indices: Dict[str, np.ndarray],
        num_rows: int,
    ) -> None:
        if not encounter_cat_indices:
            return
        cat_tensors: List[torch.Tensor] = []
        for col in self.columns.categorical_low_card:
            if col not in encounter_cat_indices:
                continue
            values = np.asarray(encounter_cat_indices[col])
            if values.shape[0] != num_rows:
                raise ValueError(f"Categorical column {col} has mismatched length")
            cat_tensors.append(torch.as_tensor(values, dtype=torch.long).view(num_rows, 1))
        if cat_tensors:
            data["encounter"].categorical = torch.cat(cat_tensors, dim=1)

    def _add_entity_nodes(self, data: HeteroData) -> None:
        for node_type, enabled in self.graph_config.node_types_enabled.dict().items():
            if node_type == "encounter" or not enabled:
                continue
            vocab = self.vocabs.get(node_type)
            if vocab is None:
                continue
            num_nodes = len(vocab.itos)
            data[node_type].num_nodes = num_nodes
            data[node_type].x = torch.arange(num_nodes, dtype=torch.long)

    def _lookup(self, node_type: str, value: str) -> int:
        vocab = self.vocabs.get(node_type)
        if vocab is None:
            raise KeyError(f"No vocabulary available for node type {node_type}")
        if value in vocab.stoi:
            return vocab.stoi[value]
        if vocab.unknown_token is not None and vocab.unknown_token in vocab.stoi:
            return vocab.stoi[vocab.unknown_token]
        raise KeyError(f"Value {value} missing from vocab {node_type}")

    def _normalise_icd(self, raw: object) -> Optional[str]:
        if pd.isna(raw):
            return None
        code = str(raw).strip()
        if not code:
            return None
        if self.preprocessing.truncate_icd_to_3_digits:
            code = code[:3]
        return code

    def _resolve_icd_group(self, truncated: str, original: object) -> Optional[str]:
        mapping = self.mappings.icd_to_group
        if not mapping:
            return None
        if truncated in mapping:
            return mapping[truncated]
        orig = str(original).strip()
        return mapping.get(orig) if orig else None

    def _add_edges(self, data: HeteroData, df: pd.DataFrame) -> None:
        toggles = self.graph_config.edge_types_enabled

        if toggles.encounter__has_icd__icd and self.columns.icd_cols:
            encounter_to_icd: List[Tuple[int, int]] = []
            icd_to_group_pairs: set[Tuple[int, int]] = set()
            for encounter_idx in range(len(df)):
                row = df.iloc[encounter_idx]
                for col in self.columns.icd_cols:
                    code = self._normalise_icd(row[col])
                    if code is None:
                        continue
                    try:
                        icd_idx = self._lookup("icd", code)
                    except KeyError:
                        continue
                    encounter_to_icd.append((encounter_idx, icd_idx))
                    if (
                        toggles.icd__is_a__icd_group
                        and self.preprocessing.map_icd_to_group
                        and "icd_group" in self.vocabs
                    ):
                        group = self._resolve_icd_group(code, row[col])
                        if group is None:
                            continue
                        try:
                            group_idx = self._lookup("icd_group", group)
                        except KeyError:
                            continue
                        icd_to_group_pairs.add((icd_idx, group_idx))
            if encounter_to_icd:
                src, dst = zip(*encounter_to_icd)
                data["encounter", "has_icd", "icd"].edge_index = torch.tensor([src, dst], dtype=torch.long)
            if icd_to_group_pairs:
                src, dst = zip(*sorted(icd_to_group_pairs))
                data["icd", "is_a", "icd_group"].edge_index = torch.tensor([src, dst], dtype=torch.long)

        if toggles.encounter__has_drug__drug and self.columns.drug_cols:
            use_subtypes = self.graph_config.edge_featureing.has_drug.relation_subtypes_by_status
            use_attr = self.graph_config.edge_featureing.has_drug.edge_attr_status
            base_edges: List[Tuple[int, int]] = []
            edge_attr: List[int] = []
            subtype_edges: Dict[str, List[Tuple[int, int]]] = {}
            drug_class_pairs: set[Tuple[int, int]] = set()
            for encounter_idx in range(len(df)):
                row = df.iloc[encounter_idx]
                for drug in self.columns.drug_cols:
                    status_raw = row.get(drug, "No")
                    status = str(status_raw).strip() if pd.notna(status_raw) else "No"
                    if status not in DRUG_STATUS_VALUES:
                        status = "No"
                    if status == "No":
                        continue
                    try:
                        drug_idx = self._lookup("drug", drug)
                    except KeyError:
                        continue
                    if use_subtypes:
                        relation = f"has_drug_{status.lower()}"
                        subtype_edges.setdefault(relation, []).append((encounter_idx, drug_idx))
                    else:
                        base_edges.append((encounter_idx, drug_idx))
                        if use_attr:
                            edge_attr.append(STATUS_TO_INDEX[status])
                    if (
                        toggles.drug__belongs_to__drug_class
                        and self.preprocessing.map_drug_to_class
                        and "drug_class" in self.vocabs
                    ):
                        drug_class = self.mappings.drug_to_class.get(drug)
                        if not drug_class:
                            continue
                        try:
                            class_idx = self._lookup("drug_class", drug_class)
                        except KeyError:
                            continue
                        drug_class_pairs.add((drug_idx, class_idx))
            if base_edges:
                src, dst = zip(*base_edges)
                data["encounter", "has_drug", "drug"].edge_index = torch.tensor([src, dst], dtype=torch.long)
                if use_attr and edge_attr:
                    data["encounter", "has_drug", "drug"].edge_attr = torch.tensor(edge_attr, dtype=torch.long).view(-1, 1)
            for relation, pairs in subtype_edges.items():
                src, dst = zip(*pairs)
                data["encounter", relation, "drug"].edge_index = torch.tensor([src, dst], dtype=torch.long)
            if drug_class_pairs:
                src, dst = zip(*sorted(drug_class_pairs))
                data["drug", "belongs_to", "drug_class"].edge_index = torch.tensor([src, dst], dtype=torch.long)

        self._add_singleton_edge(
            df,
            data,
            self.columns.hospital_col,
            "encounter",
            "at_hospital",
            "hosp",
            toggles.encounter__at_hospital__hosp,
        )
        self._add_singleton_edge(
            df,
            data,
            self.columns.specialty_col,
            "encounter",
            "has_specialty",
            "specialty",
            toggles.encounter__has_specialty__specialty,
        )
        self._add_singleton_edge(
            df,
            data,
            self.columns.admission_source_col,
            "encounter",
            "has_admission_source",
            "admission_source",
            toggles.encounter__has_admission_source__admission_source,
        )
        self._add_singleton_edge(
            df,
            data,
            self.columns.admission_type_col,
            "encounter",
            "has_admission_type",
            "admission_type",
            toggles.encounter__has_admission_type__admission_type,
        )
        self._add_singleton_edge(
            df,
            data,
            self.columns.discharge_disposition_col,
            "encounter",
            "has_discharge",
            "discharge_disposition",
            toggles.encounter__has_discharge__discharge_disposition,
        )

        if self.graph_config.edge_types_enabled.reverse_edges:
            self._add_reverse_edges(data)

    def _add_singleton_edge(
        self,
        df: pd.DataFrame,
        data: HeteroData,
        column: str,
        src_type: str,
        relation: str,
        dst_type: str,
        enabled: bool,
    ) -> None:
        if not enabled or column not in df.columns:
            return
        if dst_type not in self.vocabs:
            return
        pairs: List[Tuple[int, int]] = []
        for encounter_idx in range(len(df)):
            value = df.iloc[encounter_idx][column]
            if pd.isna(value):
                continue
            try:
                dst_idx = self._lookup(dst_type, str(value))
            except KeyError:
                continue
            pairs.append((encounter_idx, dst_idx))
        if not pairs:
            return
        src, dst = zip(*pairs)
        data[src_type, relation, dst_type].edge_index = torch.tensor([src, dst], dtype=torch.long)

    def _add_reverse_edges(self, data: HeteroData) -> None:
        for edge_type in list(data.edge_types):
            edge_index = data[edge_type].edge_index
            if edge_index.numel() == 0:
                continue
            src_type, relation, dst_type = edge_type
            reverse_type = (dst_type, f"rev_{relation}", src_type)
            if reverse_type in data.edge_types:
                continue
            data[reverse_type].edge_index = edge_index.flip(0)


__all__ = ["GraphBuilder"]
