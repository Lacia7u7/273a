"""Inductive star-subgraph construction for new encounters."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ..data.mappings import HierarchyMappings
from ..data.vocab import Vocab
from ..utils.config import ColumnsConfig, GraphConfig, PreprocessingConfig

DRUG_STATUS_VALUES = {"Up", "Down", "Steady", "No"}
STATUS_TO_INDEX = {"Up": 0, "Down": 1, "Steady": 2, "No": 3}


class StarGraphBuilder:
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

    def build_from_row(
        self,
        row: pd.Series,
        encounter_features: np.ndarray,
        encounter_cat_indices: Dict[str, np.ndarray],
    ) -> HeteroData:
        features = np.asarray(encounter_features).reshape(1, -1)
        data = HeteroData()
        data["encounter"].num_nodes = 1
        data["encounter"].x = torch.as_tensor(features, dtype=torch.float32)
        if encounter_cat_indices:
            cat_tensors: List[torch.Tensor] = []
            for col in self.columns.categorical_low_card:
                if col not in encounter_cat_indices:
                    continue
                values = np.asarray(encounter_cat_indices[col]).reshape(1, -1)
                cat_tensors.append(torch.as_tensor(values, dtype=torch.long))
            if cat_tensors:
                data["encounter"].categorical = torch.cat(cat_tensors, dim=1)

        for node_type, vocab in self.vocabs.items():
            if node_type == "encounter":
                continue
            data[node_type].num_nodes = len(vocab.itos)
            data[node_type].x = torch.arange(len(vocab.itos), dtype=torch.long)

        self._add_edges(data, row)

        if self.graph_config.edge_types_enabled.reverse_edges:
            for edge_type in list(data.edge_types):
                edge_index = data[edge_type].edge_index
                if edge_index.numel() == 0:
                    continue
                src, rel, dst = edge_type
                reverse_type = (dst, f"rev_{rel}", src)
                if reverse_type in data.edge_types:
                    continue
                data[reverse_type].edge_index = edge_index.flip(0)
        return data

    def _lookup(self, node_type: str, value: str) -> int:
        vocab = self.vocabs.get(node_type)
        if vocab is None:
            raise KeyError(f"No vocabulary for node type {node_type}")
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

    def _add_edges(self, data: HeteroData, row: pd.Series) -> None:
        toggles = self.graph_config.edge_types_enabled

        if toggles.encounter__has_icd__icd and self.columns.icd_cols:
            encounter_to_icd: List[int] = []
            icd_groups: set[tuple[int, int]] = set()
            for col in self.columns.icd_cols:
                code = self._normalise_icd(row.get(col))
                if code is None:
                    continue
                try:
                    icd_idx = self._lookup("icd", code)
                except KeyError:
                    continue
                encounter_to_icd.append(icd_idx)
                if (
                    toggles.icd__is_a__icd_group
                    and self.preprocessing.map_icd_to_group
                    and "icd_group" in self.vocabs
                ):
                    group = self._resolve_icd_group(code, row.get(col))
                    if group is None:
                        continue
                    try:
                        group_idx = self._lookup("icd_group", group)
                    except KeyError:
                        continue
                    icd_groups.add((icd_idx, group_idx))
            if encounter_to_icd:
                src = torch.zeros(len(encounter_to_icd), dtype=torch.long)
                dst = torch.tensor(encounter_to_icd, dtype=torch.long)
                data["encounter", "has_icd", "icd"].edge_index = torch.stack([src, dst], dim=0)
            if icd_groups:
                src, dst = zip(*sorted(icd_groups))
                data["icd", "is_a", "icd_group"].edge_index = torch.tensor([src, dst], dtype=torch.long)

        if toggles.encounter__has_drug__drug and self.columns.drug_cols:
            use_subtypes = self.graph_config.edge_featureing.has_drug.relation_subtypes_by_status
            use_attr = self.graph_config.edge_featureing.has_drug.edge_attr_status
            base_edges: List[int] = []
            edge_attr: List[int] = []
            subtype_edges: Dict[str, List[int]] = {}
            drug_classes: set[tuple[int, int]] = set()
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
                    subtype_edges.setdefault(relation, []).append(drug_idx)
                else:
                    base_edges.append(drug_idx)
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
                    drug_classes.add((drug_idx, class_idx))
            if base_edges:
                src = torch.zeros(len(base_edges), dtype=torch.long)
                dst = torch.tensor(base_edges, dtype=torch.long)
                data["encounter", "has_drug", "drug"].edge_index = torch.stack([src, dst], dim=0)
                if use_attr and edge_attr:
                    data["encounter", "has_drug", "drug"].edge_attr = torch.tensor(edge_attr, dtype=torch.long).view(-1, 1)
            for relation, indices in subtype_edges.items():
                src = torch.zeros(len(indices), dtype=torch.long)
                dst = torch.tensor(indices, dtype=torch.long)
                data["encounter", relation, "drug"].edge_index = torch.stack([src, dst], dim=0)
            if drug_classes:
                src, dst = zip(*sorted(drug_classes))
                data["drug", "belongs_to", "drug_class"].edge_index = torch.tensor([src, dst], dtype=torch.long)

        self._add_singleton_edge(row, data, self.columns.hospital_col, "encounter", "at_hospital", "hosp", toggles.encounter__at_hospital__hosp)
        self._add_singleton_edge(
            row,
            data,
            self.columns.specialty_col,
            "encounter",
            "has_specialty",
            "specialty",
            toggles.encounter__has_specialty__specialty,
        )
        self._add_singleton_edge(
            row,
            data,
            self.columns.admission_source_col,
            "encounter",
            "has_admission_source",
            "admission_source",
            toggles.encounter__has_admission_source__admission_source,
        )
        self._add_singleton_edge(
            row,
            data,
            self.columns.admission_type_col,
            "encounter",
            "has_admission_type",
            "admission_type",
            toggles.encounter__has_admission_type__admission_type,
        )
        self._add_singleton_edge(
            row,
            data,
            self.columns.discharge_disposition_col,
            "encounter",
            "has_discharge",
            "discharge_disposition",
            toggles.encounter__has_discharge__discharge_disposition,
        )

    def _add_singleton_edge(
        self,
        row: pd.Series,
        data: HeteroData,
        column: str,
        src_type: str,
        relation: str,
        dst_type: str,
        enabled: bool,
    ) -> None:
        if not enabled or column not in row.index:
            return
        value = row.get(column)
        if pd.isna(value) or dst_type not in self.vocabs:
            return
        try:
            dst_idx = self._lookup(dst_type, str(value))
        except KeyError:
            return
        edge_index = torch.tensor([[0], [dst_idx]], dtype=torch.long)
        data[src_type, relation, dst_type].edge_index = edge_index


__all__ = ["StarGraphBuilder"]
