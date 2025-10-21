"""Neighbor sampling utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from torch_geometric.loader import HGTLoader


def build_relation_fanouts(fanouts: Dict[str, List[int]]) -> Dict[str, List[int]]:
    parsed: Dict[Tuple[str, str, str], List[int]] = {}
    for key, values in fanouts.items():
        parts = key.split("__")
        if len(parts) != 3:
            raise ValueError(f"Relation key '{key}' must follow 'src__rel__dst' pattern")
        parsed[(parts[0], parts[1], parts[2])] = list(values)
    return parsed


def create_hgt_loader(hetero_data, input_nodes, fanouts: Dict[str, List[int]], batch_size: int, shuffle: bool = True):
    """Create a :class:`torch_geometric.loader.HGTLoader` with relation-aware fanouts."""
    num_samples = build_relation_fanouts(fanouts)
    return HGTLoader(
        hetero_data,
        num_samples=num_samples,
        shuffle=shuffle,
        input_nodes=input_nodes,
        batch_size=batch_size,
    )


__all__ = ["build_relation_fanouts", "create_hgt_loader"]
