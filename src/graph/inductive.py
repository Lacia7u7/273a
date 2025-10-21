"""Inductive star-subgraph construction for new encounters."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ..data.mappings import HierarchyMappings
from ..data.vocab import Vocab
from ..utils.config import ColumnsConfig, GraphConfig, PreprocessingConfig


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
        data = HeteroData()
        data["encounter"].x = torch.tensor(encounter_features.reshape(1, -1), dtype=torch.float32)
        if encounter_cat_indices:
            cat = [torch.tensor(v.reshape(1, -1), dtype=torch.long) for v in encounter_cat_indices.values()]
            data["encounter"].categorical = torch.cat(cat, dim=1)
        for node_type, vocab in self.vocabs.items():
            if node_type == "encounter":
                continue
            data[node_type].num_nodes = len(vocab.itos)
            data[node_type].x = torch.arange(len(vocab.itos), dtype=torch.long)
        self._add_edges(data, row)
        if self.graph_config.edge_types_enabled.reverse_edges:
            for edge_type in list(data.edge_types):
                src, rel, dst = edge_type
                rev_edge = data[edge_type].edge_index.flip(0)
                data[dst, f"rev_{rel}", src].edge_index = rev_edge
        return data

    def _lookup(self, node_type: str, value: str) -> int:
        vocab = self.vocabs[node_type]
        if value in vocab.stoi:
            return vocab.stoi[value]
        if vocab.unknown_token is not None:
            return vocab.stoi[vocab.unknown_token]
        raise KeyError(f"Value {value} missing from vocab {node_type}")

    def _add_edges(self, data: HeteroData, row: pd.Series) -> None:
        def add_edge(src_idx: int, dst_type: str, relation: str, value: str) -> None:
            dst_idx = self._lookup(dst_type, value)
            data["encounter", relation, dst_type].edge_index = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)

        for col in self.columns.icd_cols:
            code = str(row[col]) if pd.notna(row[col]) else ""
            if not code:
                continue
            if self.preprocessing.truncate_icd_to_3_digits:
                code = code[:3]
            try:
                dst = self._lookup("icd", code)
            except KeyError:
                dst = self._lookup("icd", self.vocabs["icd"].unknown_token)
            idx = torch.tensor([[0], [dst]], dtype=torch.long)
            data["encounter", "has_icd", "icd"].edge_index = (
                torch.cat([data["encounter", "has_icd", "icd"].edge_index, idx], dim=1)
                if ("encounter", "has_icd", "icd") in data.edge_types
                else idx
            )

        for drug in self.columns.drug_cols:
            status = str(row[drug]) if pd.notna(row[drug]) else "No"
            if status == "No":
                continue
            try:
                dst = self._lookup("drug", drug)
            except KeyError:
                dst = self._lookup("drug", self.vocabs["drug"].unknown_token)
            idx = torch.tensor([[0], [dst]], dtype=torch.long)
            data["encounter", "has_drug", "drug"].edge_index = (
                torch.cat([data["encounter", "has_drug", "drug"].edge_index, idx], dim=1)
                if ("encounter", "has_drug", "drug") in data.edge_types
                else idx
            )

        if self.columns.hospital_col in row and pd.notna(row[self.columns.hospital_col]):
            value = str(row[self.columns.hospital_col])
            add_edge(0, "hosp", "at_hospital", value)
        if self.columns.specialty_col in row and pd.notna(row[self.columns.specialty_col]):
            add_edge(0, "specialty", "has_specialty", str(row[self.columns.specialty_col]))
        if self.columns.admission_source_col in row and pd.notna(row[self.columns.admission_source_col]):
            add_edge(0, "admission_source", "has_admission_source", str(row[self.columns.admission_source_col]))
        if self.columns.admission_type_col in row and pd.notna(row[self.columns.admission_type_col]):
            add_edge(0, "admission_type", "has_admission_type", str(row[self.columns.admission_type_col]))
        if self.columns.discharge_disposition_col in row and pd.notna(row[self.columns.discharge_disposition_col]):
            add_edge(0, "discharge_disposition", "has_discharge", str(row[self.columns.discharge_disposition_col]))


__all__ = ["StarGraphBuilder"]
