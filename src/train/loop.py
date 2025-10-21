"""Training loop with early stopping and TensorBoard logging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from .metrics import compute_metrics


@dataclass
class TrainState:
    epoch: int
    best_metric: float
    best_state_dict: Dict[str, torch.Tensor]
    best_optimizer_state: Dict[str, torch.Tensor]


class EarlyStopper:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = -np.inf if mode == "max" else np.inf

    def step(self, metric: float) -> bool:
        improved = metric > self.best if self.mode == "max" else metric < self.best
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter > self.patience


def train_epochs(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_loader,
    val_loader,
    writer: SummaryWriter,
    max_epochs: int,
    device: torch.device,
    early_stopping_patience: int,
    monitor_metric: str,
    gradient_clip_norm: Optional[float] = None,
) -> TrainState:
    model.to(device)
    loss_fn.to(device)
    early_stopper = EarlyStopper(patience=early_stopping_patience, mode="max")
    best_state = None
    best_opt = None
    best_metric = -np.inf
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            target = batch["encounter"].y.float().to(device)
            loss = loss_fn(logits, target)
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar("train/loss", running_loss / max(1, len(train_loader)), epoch)

        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                target = batch["encounter"].y.float().to(device)
                all_logits.append(logits.cpu())
                all_targets.append(target.cpu())
        logits = torch.cat(all_logits).numpy()
        targets = torch.cat(all_targets).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        metrics = compute_metrics(targets, probs)
        writer.add_scalar(f"val/{monitor_metric}", metrics[monitor_metric], epoch)

        if metrics[monitor_metric] > best_metric:
            best_metric = metrics[monitor_metric]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_opt = optimizer.state_dict()

        if early_stopper.step(metrics[monitor_metric]):
            break

    assert best_state is not None and best_opt is not None
    return TrainState(epoch=epoch, best_metric=best_metric, best_state_dict=best_state, best_optimizer_state=best_opt)


__all__ = ["train_epochs", "TrainState"]
