from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _safe_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    if tensor.is_cuda and device.type == "cuda" and tensor.device.index != device.index:
        return tensor.to(torch.device("cpu")).to(device)
    return tensor.to(device, non_blocking=True)


def mask_mse_loss(x_rec: torch.Tensor, x_target: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss over masked entries only."""

    if x_target.device != x_rec.device:
        x_target = _safe_to_device(x_target, x_rec.device)

    if mask_idx.device != x_rec.device:
        mask_idx = _safe_to_device(mask_idx, x_rec.device)
    if mask_idx.dtype != torch.bool:
        mask_idx = mask_idx.bool()

    if mask_idx.numel() == 0 or int(mask_idx.sum().item()) == 0:
        return torch.zeros((), dtype=x_rec.dtype, device=x_rec.device)

    return F.mse_loss(x_rec[mask_idx], x_target[mask_idx])


def cluster_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy loss for pseudo-label cluster supervision."""

    if labels.device != logits.device:
        labels = _safe_to_device(labels, logits.device)
    if class_weights is not None:
        class_weights = _safe_to_device(class_weights, logits.device).to(dtype=logits.dtype)
    return F.cross_entropy(logits, labels, weight=class_weights)


def smoothness_loss(z: torch.Tensor, edge_index: torch.Tensor, mode: str = "l2") -> torch.Tensor:
    """Neighbor smoothness regularizer over graph edges."""

    if edge_index.numel() == 0:
        return torch.zeros((), dtype=z.dtype, device=z.device)

    if edge_index.device != z.device:
        edge_index = _safe_to_device(edge_index, z.device)

    src = edge_index[0]
    dst = edge_index[1]
    diff = z[src] - z[dst]

    if mode == "l2":
        return diff.pow(2).sum(dim=-1).mean()
    if mode == "l1":
        return diff.abs().sum(dim=-1).mean()
    raise ValueError(f"Unknown smoothness mode: {mode}")


def consistency_loss(z1: torch.Tensor, z2: torch.Tensor, mode: str = "mse") -> torch.Tensor:
    """Consistency loss between two stochastic embedding passes."""

    if z2.device != z1.device:
        z2 = _safe_to_device(z2, z1.device)

    if mode == "mse":
        return F.mse_loss(z1, z2)
    if mode == "l2":
        return (z1 - z2).pow(2).sum(dim=-1).mean()
    raise ValueError(f"Unknown consistency mode: {mode}")
