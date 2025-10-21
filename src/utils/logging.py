"""Logging utilities providing a unified TensorBoard writer and Python logging."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
    )


def create_tensorboard_writer(logdir: str, run_id: Optional[str] = None) -> SummaryWriter:
    path = Path(logdir)
    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(path if run_id is None else path / run_id))


__all__ = ["configure_logging", "create_tensorboard_writer"]
