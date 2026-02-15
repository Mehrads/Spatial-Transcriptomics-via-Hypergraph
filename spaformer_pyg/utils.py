from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and Torch seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch versions may not expose this API.
            pass


def reorder_labels_top_to_bottom(labels: Iterable[int], coords_y: Iterable[float]) -> np.ndarray:
    """Relabel clusters by increasing mean y-coordinate.

    Noise label `-1` is preserved.
    """

    labels = np.asarray(labels, dtype=int)
    coords_y = np.asarray(coords_y, dtype=float)

    if labels.shape[0] != coords_y.shape[0]:
        raise ValueError("labels and coords_y must have the same length")

    valid = labels != -1
    unique = np.unique(labels[valid])

    ordered = sorted(unique, key=lambda lab: float(coords_y[labels == lab].mean()))
    mapping = {old: idx for idx, old in enumerate(ordered)}

    remapped = np.full(labels.shape, -1, dtype=int)
    for old, new in mapping.items():
        remapped[labels == old] = new
    return remapped


def get_device(preference: str = "auto") -> torch.device:
    """Pick CUDA when available unless CPU is explicitly requested."""

    pref = str(preference).lower().strip()

    if pref == "cpu":
        return torch.device("cpu")

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if pref != "auto":
        raise ValueError(f"Unknown device preference: {preference}")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
