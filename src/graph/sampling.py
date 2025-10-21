"""Neighbor sampling utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List

from torch_geometric.loader import HGTLoader


def build_relation_fanouts(fanouts: Dict[str, List[int]]) -> Dict[str, List[int]]:
    return fanouts


def create_hgt_loader(hetero_data, input_nodes, fanouts: Dict[str, List[int]], batch_size: int, shuffle: bool = True):
    """Create a :class:`torch_geometric.loader.HGTLoader` with relation-aware fanouts."""
    return HGTLoader(
        hetero_data,
        num_samples=fanouts,
        shuffle=shuffle,
        input_nodes=input_nodes,
        batch_size=batch_size,
    )


__all__ = ["build_relation_fanouts", "create_hgt_loader"]
