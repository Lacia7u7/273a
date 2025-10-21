"""Vocabulary management for categorical entities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    unknown_token: Optional[str] = None

    def lookup(self, value: str) -> int:
        key = value if value in self.stoi else self.unknown_token
        if key is None:
            raise KeyError(f"Value '{value}' not in vocabulary and no unknown token defined")
        return self.stoi[key]

    def to_dict(self) -> Dict[str, List[str]]:
        return {"itos": self.itos, "unknown_token": self.unknown_token}

    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        min_freq: int = 1,
        unknown_token: Optional[str] = None,
    ) -> "Vocab":
        counts = series.value_counts()
        tokens = counts[counts >= min_freq].index.tolist()
        if unknown_token and unknown_token not in tokens:
            tokens.append(unknown_token)
        itos = sorted(tokens)
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos, unknown_token=unknown_token)

    @classmethod
    def from_dict(cls, payload: Mapping[str, List[str]]) -> "Vocab":
        itos = list(payload["itos"])
        unknown_token = payload.get("unknown_token")
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos, unknown_token=unknown_token)


def build_vocab_from_iterable(
    values: Iterable[str],
    min_freq: int,
    unknown_token: Optional[str] = None,
) -> Vocab:
    series = pd.Series(list(values))
    return Vocab.from_series(series, min_freq=min_freq, unknown_token=unknown_token)


__all__ = ["Vocab", "build_vocab_from_iterable"]
