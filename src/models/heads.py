"""Model heads for encounter classification."""
from __future__ import annotations

import torch
from torch import nn


class EncounterHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.linear(self.dropout(x)).squeeze(-1)


__all__ = ["EncounterHead"]
