"""Split utilities with leakage prevention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ..utils.config import SplitStrategy


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray


def _get_groups(df: pd.DataFrame, group_by: str) -> np.ndarray:
    return df[group_by].to_numpy()


def make_splits(df: pd.DataFrame, strategy: SplitStrategy, group_column: str, target: np.ndarray) -> List[SplitIndices]:
    groups = _get_groups(df, group_column)
    splits: List[SplitIndices] = []
    if strategy.strategy == "group_k_fold":
        splitter = GroupKFold(n_splits=strategy.n_splits)
        for train_idx, val_idx in splitter.split(df, target, groups):
            splits.append(SplitIndices(train=train_idx, val=val_idx))
    elif strategy.strategy == "group_shuffle":
        splitter = GroupShuffleSplit(n_splits=strategy.n_splits, random_state=strategy.seed, test_size=1 / strategy.n_splits)
        for train_idx, val_idx in splitter.split(df, target, groups):
            splits.append(SplitIndices(train=train_idx, val=val_idx))
    else:
        raise ValueError(f"Unsupported split strategy: {strategy.strategy}")
    return splits


def assert_no_group_leakage(df: pd.DataFrame, splits: List[SplitIndices], group_column: str) -> None:
    seen_sets: List[set] = []
    for split in splits:
        train_groups = set(df.iloc[split.train][group_column].unique())
        val_groups = set(df.iloc[split.val][group_column].unique())
        assert train_groups.isdisjoint(val_groups), "Leakage detected between train and validation groups"
        seen_sets.append(train_groups)
        seen_sets.append(val_groups)

    # ensure global disjointness
    all_seen = [groups for groups in seen_sets if groups]
    for i, groups_a in enumerate(all_seen):
        for groups_b in all_seen[i + 1 :]:
            assert groups_a.isdisjoint(groups_b), "Group leakage detected across splits"


__all__ = ["make_splits", "assert_no_group_leakage", "SplitIndices"]
