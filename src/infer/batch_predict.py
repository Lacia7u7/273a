"""Batch inductive inference utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from ..graph.inductive import StarGraphBuilder
from ..utils.io import load_pickle


def batch_predict(
    model: torch.nn.Module,
    preprocessor,
    star_builder: StarGraphBuilder,
    csv_path: str,
    output_path: str,
    device: torch.device,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    predictions = []
    for _, row in df.iterrows():
        numeric, _, cat_indices = preprocessor.transform(pd.DataFrame([row]))
        graph = star_builder.build_from_row(row, numeric[0], cat_indices)
        graph = graph.to(device)
        with torch.no_grad():
            logits = model(graph)
            prob = torch.sigmoid(logits).cpu().item()
        predictions.append({"encounter_id": row.get("encounter_id", _), "prob": prob})
    result = pd.DataFrame(predictions)
    result["y_hat"] = (result["prob"] >= 0.5).astype(int)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


__all__ = ["batch_predict"]
