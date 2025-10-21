"""Heterogeneous Graph Transformer model."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch_geometric.nn import HGTConv

from .heads import EncounterHead


class HGTModel(nn.Module):
    def __init__(
        self,
        metadata: Tuple,
        num_nodes_dict: Dict[str, int],
        encounter_input_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        embedding_dims: Dict[str, int],
    ) -> None:
        super().__init__()
        node_types, _ = metadata
        self.embeddings = nn.ModuleDict()
        self.node_projections = nn.ModuleDict()
        for node_type in node_types:
            if node_type == "encounter":
                self.node_projections[node_type] = nn.Linear(encounter_input_dim, hidden_dim)
            else:
                num_nodes = num_nodes_dict.get(node_type, 1)
                emb_dim = embedding_dims.get(node_type, hidden_dim)
                self.embeddings[node_type] = nn.Embedding(num_nodes, emb_dim)
                self.node_projections[node_type] = nn.Linear(emb_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=heads))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.head = EncounterHead(hidden_dim, dropout)

    def forward(self, data) -> torch.Tensor:  # type: ignore
        x_dict = {}
        for node_type, x in data.x_dict.items():
            if node_type == "encounter":
                x_dict[node_type] = self.node_projections[node_type](x.float())
            else:
                indices = x.squeeze(-1).long() if x.dim() > 1 else x.long()
                emb = self.embeddings[node_type](indices)
                x_dict[node_type] = self.node_projections[node_type](emb)

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = self.dropout(self.activation(x_dict[node_type]))

        logits = self.head(x_dict["encounter"])
        return logits


__all__ = ["HGTModel"]
