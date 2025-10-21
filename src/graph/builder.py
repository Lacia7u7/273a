"""Build :class:`torch_geometric.data.HeteroData` objects from tabular data."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from ..data.mappings import HierarchyMappings
from ..data.vocab import Vocab
from ..utils.config import ColumnsConfig, GraphConfig, PreprocessingConfig

DRUG_STATUS_VALUES = {"Up", "Down", "Steady", "No"}


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
        data = HeteroData()
        data["encounter"].x = torch.tensor(encounter_features, dtype=torch.float32)
        data["encounter"].y = torch.tensor(df["target"].values, dtype=torch.float32)
        data["encounter"].encounter_id = torch.tensor(df[self.columns.hospital_col].index.values, dtype=torch.long)

        self._add_encounter_categorical_features(data, encounter_cat_indices)
        self._add_entity_nodes(data)
        self._add_edges(data, df)
        return data

    def _add_encounter_categorical_features(
        self, data: HeteroData, encounter_cat_indices: Dict[str, np.ndarray]
    ) -> None:
        if not encounter_cat_indices:
            return
        cat_features = []
        for col in self.columns.categorical_low_card:
            if col in encounter_cat_indices:
                cat_features.append(torch.tensor(encounter_cat_indices[col], dtype=torch.long).unsqueeze(-1))
        if cat_features:
            data["encounter"].categorical = torch.cat(cat_features, dim=1)

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
        vocab = self.vocabs[node_type]
        if value in vocab.stoi:
            return vocab.stoi[value]
        if vocab.unknown_token is not None:
            return vocab.stoi[vocab.unknown_token]
        raise KeyError(f"Value {value} missing from vocab {node_type}")

    def _add_edges(self, data: HeteroData, df: pd.DataFrame) -> None:
        columns = self.columns
        gc = self.graph_config
        id_cols = {
            "icd": columns.icd_cols,
            "drug": columns.drug_cols,
        }

        if gc.edge_types_enabled.encounter__has_icd__icd:
            src, dst = [], []
            group_edges: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
            for idx, row in df.iterrows():
                encounter_idx = idx
                for col in columns.icd_cols:
                    code = str(row[col]) if pd.notna(row[col]) else ""
                    if not code:
                        continue
                    if self.preprocessing.truncate_icd_to_3_digits:
                        code = code[:3]
                    icd_index = self._lookup("icd", code)
                    src.append(encounter_idx)
                    dst.append(icd_index)
                    if gc.edge_types_enabled.icd__is_a__icd_group and self.preprocessing.map_icd_to_group:
                        group = self.mappings.icd_to_group.get(code)
                        if group:
                            group_idx = self._lookup("icd_group", group)
                            group_edges[group].append((icd_index, group_idx))
            if src:
                edge_index = torch.tensor([src, dst], dtype=torch.long)
                data["encounter", "has_icd", "icd"].edge_index = edge_index.t().t()
            if group_edges:
                rel_src, rel_dst = zip(*[pair for pairs in group_edges.values() for pair in pairs])
                edge_index = torch.tensor([rel_src, rel_dst], dtype=torch.long)
                data["icd", "is_a", "icd_group"].edge_index = edge_index

        if gc.edge_types_enabled.encounter__has_drug__drug:
            drug_edges = defaultdict(list)
            status_mapping = {"Up": 0, "Down": 1, "Steady": 2, "No": 3}
            for idx, row in df.iterrows():
                encounter_idx = idx
                for drug in columns.drug_cols:
                    status = str(row[drug]) if pd.notna(row[drug]) else "No"
                    if status not in DRUG_STATUS_VALUES:
                        status = "No"
                    if status == "No":
                        continue
                    drug_idx = self._lookup("drug", drug)
                    drug_edges["base"].append((encounter_idx, drug_idx, status_mapping[status]))
                    if gc.edge_types_enabled.drug__belongs_to__drug_class and self.preprocessing.map_drug_to_class:
                        drug_class = self.mappings.drug_to_class.get(drug)
                        if drug_class:
                            class_idx = self._lookup("drug_class", drug_class)
                            drug_edges["class"].append((drug_idx, class_idx, 0))
            if drug_edges["base"]:
                src, dst, status = zip(*drug_edges["base"])
                data["encounter", "has_drug", "drug"].edge_index = torch.tensor([src, dst], dtype=torch.long)
                if gc.edge_featureing.has_drug.edge_attr_status:
                    data["encounter", "has_drug", "drug"].edge_attr = torch.tensor(status, dtype=torch.long).unsqueeze(-1)
            if drug_edges["class"]:
                src, dst, _ = zip(*drug_edges["class"])
                data["drug", "belongs_to", "drug_class"].edge_index = torch.tensor([src, dst], dtype=torch.long)

        self._add_singleton_edge(df, data, columns.hospital_col, "encounter", "at_hospital", "hosp")
        self._add_singleton_edge(df, data, columns.specialty_col, "encounter", "has_specialty", "specialty")
        self._add_singleton_edge(df, data, columns.admission_source_col, "encounter", "has_admission_source", "admission_source")
        self._add_singleton_edge(df, data, columns.admission_type_col, "encounter", "has_admission_type", "admission_type")
        self._add_singleton_edge(df, data, columns.discharge_disposition_col, "encounter", "has_discharge", "discharge_disposition")

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
    ) -> None:
        if column not in df.columns:
            return
        if (src_type, relation, dst_type) in data.edge_types:
            return
        if dst_type not in self.vocabs:
            return
        src, dst = [], []
        for idx, value in enumerate(df[column]):
            if pd.isna(value):
                continue
            try:
                dst_idx = self._lookup(dst_type, str(value))
            except KeyError:
                continue
            src.append(idx)
            dst.append(dst_idx)
        if src:
            data[src_type, relation, dst_type].edge_index = torch.tensor([src, dst], dtype=torch.long)

    def _add_reverse_edges(self, data: HeteroData) -> None:
        for edge_type in list(data.edge_types):
            src_type, relation, dst_type = edge_type
            rev_relation = f"rev_{relation}"
            edge_index = data[edge_type].edge_index
            rev_edge_index = edge_index.flip(0)
            data[dst_type, rev_relation, src_type].edge_index = rev_edge_index


__all__ = ["GraphBuilder"]
