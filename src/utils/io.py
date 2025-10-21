"""Artifact persistence helpers."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import torch


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_torch(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_torch(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


__all__ = [
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "save_torch",
    "load_torch",
]
