from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


def _normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape={tuple(edge_index.shape)}")
    if edge_index.shape[0] == 2:
        out = edge_index
    elif edge_index.shape[1] == 2:
        out = edge_index.t().contiguous()
    else:
        raise ValueError(f"edge_index must have shape [2, E] or [E, 2], got {tuple(edge_index.shape)}")
    return out.long().contiguous()


def load_prepared_data(
    data_dir: str | Path = "Data/spaformer_prepared",
) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[Dict[str, torch.FloatTensor]]]:
    """Load SpaFormer-ready arrays and return tensors for PyG use.

    Returns
    -------
    x
        Float32 tensor of shape [N, F].
    edge_index
        Int64 COO tensor of shape [2, E].
    coords
        Optional dict with keys from {"C", "C_raw"} if those files exist.
    """

    base = Path(data_dir)
    x_path = base / "X.npy"
    edge_path = base / "edges.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing expression matrix: {x_path}")
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing edge index: {edge_path}")

    x_np = np.load(x_path).astype(np.float32, copy=False)
    edge_np = np.load(edge_path).astype(np.int64, copy=False)

    x = torch.from_numpy(x_np)
    edge_index = _normalize_edge_index(torch.from_numpy(edge_np))

    coords: Dict[str, torch.FloatTensor] = {}
    c_path = base / "C.npy"
    c_raw_path = base / "C_raw.npy"

    if c_path.exists():
        coords["C"] = torch.from_numpy(np.load(c_path).astype(np.float32, copy=False))
    if c_raw_path.exists():
        coords["C_raw"] = torch.from_numpy(np.load(c_raw_path).astype(np.float32, copy=False))

    return x, edge_index, (coords if coords else None)


def build_pyg_data(x: torch.Tensor, edge_index: torch.Tensor) -> Data:
    """Build a PyG Data object and append self-loops."""

    if x.ndim != 2:
        raise ValueError(f"x must be 2D [N, F], got shape={tuple(x.shape)}")

    edge_index = _normalize_edge_index(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
    return Data(x=x.float(), edge_index=edge_index.long())


def to_device(data: Data, device: torch.device | str) -> Data:
    """Move a PyG Data object to a device."""

    return data.to(device)
