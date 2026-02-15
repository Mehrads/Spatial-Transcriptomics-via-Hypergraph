from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


def apply_feature_mask(
    x: torch.Tensor,
    mask_rate: float,
    mask_token: str = "zeros",
    mask_mode: str = "entry",
    mask_token_param: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply feature masking used by the reconstruction objective."""

    if x.ndim != 2:
        raise ValueError(f"x must be [N, F], got {tuple(x.shape)}")
    if not 0.0 <= float(mask_rate) <= 1.0:
        raise ValueError(f"mask_rate must be in [0, 1], got {mask_rate}")

    x_target = x.detach().clone()
    x_masked = x.clone()

    if float(mask_rate) == 0.0:
        mask_idx = torch.zeros_like(x, dtype=torch.bool)
        return x_masked, mask_idx, x_target

    if mask_mode == "entry":
        mask_idx = torch.rand(x.shape, device=x.device) < float(mask_rate)
    elif mask_mode == "node":
        node_mask = torch.rand(x.shape[0], device=x.device) < float(mask_rate)
        mask_idx = node_mask.unsqueeze(1).expand_as(x)
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    if mask_token == "zeros":
        x_masked = x_masked.masked_fill(mask_idx, 0.0)
    elif mask_token == "learnable":
        if mask_token_param is None:
            raise ValueError("mask_token_param is required when mask_token='learnable'")
        token = mask_token_param.to(device=x.device, dtype=x.dtype).view(1, -1)
        x_masked[mask_idx] = token.expand_as(x_masked)[mask_idx]
    else:
        raise ValueError(f"Unknown mask_token: {mask_token}")

    return x_masked, mask_idx, x_target


class PreModelPyG(nn.Module):
    """PyG SpaFormer-style pretraining model (GAT encoder + MLP decoder).

    Supports optional model-parallel execution by placing encoder layers across
    multiple CUDA devices.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        encoder: str = "gat",
        decoder: str = "mlp",
        heads: int = 4,
        activation: str = "prelu",
        mask_rate: float = 0.3,
        mask_token: str = "zeros",
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if encoder.lower() != "gat":
            raise ValueError(f"Only encoder='gat' is currently supported, got {encoder}")
        if decoder.lower() != "mlp":
            raise ValueError(f"Only decoder='mlp' is currently supported, got {decoder}")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.heads = int(heads)
        self.mask_rate = float(mask_rate)
        self.mask_token_type = str(mask_token)
        self.activation_name = str(activation).lower()

        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.encoder_acts = nn.ModuleList()

        prev_dim = self.in_dim
        for i in range(self.num_layers):
            self.encoder_layers.append(
                GATConv(
                    in_channels=prev_dim,
                    out_channels=self.hidden_dim,
                    heads=self.heads,
                    concat=False,
                    dropout=self.dropout,
                    add_self_loops=False,
                )
            )
            self.encoder_norms.append(nn.LayerNorm(self.hidden_dim))
            if i < self.num_layers - 1:
                self.encoder_acts.append(self._build_activation(self.activation_name))
            prev_dim = self.hidden_dim

        self.decoder_fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_fc2 = nn.Linear(self.hidden_dim, self.in_dim)
        self.decoder_act = self._build_activation(self.activation_name)

        self.mask_token_param: Optional[nn.Parameter]
        if self.mask_token_type == "learnable":
            self.mask_token_param = nn.Parameter(torch.zeros(self.in_dim))
        else:
            self.mask_token_param = None

        self.cluster_head: Optional[nn.Linear] = None

        self.model_parallel = False
        self.parallel_devices: list[torch.device] = []
        self.layer_devices: list[torch.device] = []
        self.primary_device = torch.device("cpu")
        self.output_device = torch.device("cpu")
        self._edge_index_cache: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}

    def _build_activation(self, activation: str) -> nn.Module:
        if activation == "prelu":
            return nn.PReLU(self.hidden_dim)
        if activation == "elu":
            return nn.ELU()
        raise ValueError(f"Unknown activation: {activation}")

    @staticmethod
    def _safe_tensor_to(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Move tensors safely across GPUs on hosts with broken peer copies."""
        if tensor.device == device:
            return tensor
        if tensor.is_cuda and device.type == "cuda" and tensor.device.index != device.index:
            return tensor.to(torch.device("cpu")).to(device)
        return tensor.to(device, non_blocking=True)

    @staticmethod
    def _safe_module_to(module: nn.Module, device: torch.device) -> nn.Module:
        param = next(module.parameters(), None)
        if param is not None and param.device == device:
            return module
        if param is not None and param.is_cuda and device.type == "cuda" and param.device.index != device.index:
            module = module.to(torch.device("cpu"))
            return module.to(device)
        return module.to(device)

    def set_device_layout(self, devices: Optional[Sequence[torch.device | str]] = None):
        """Configure single-device or model-parallel layer placement."""

        if not devices:
            devices = [next(self.parameters()).device]

        device_list = [torch.device(d) for d in devices]
        if len(device_list) == 0:
            device_list = [torch.device("cpu")]

        self._edge_index_cache = {}

        if len(device_list) == 1:
            dev = device_list[0]
            # Route through CPU to avoid direct GPU->GPU parameter copies.
            self.to(torch.device("cpu"))
            self.to(dev)
            self.model_parallel = False
            self.parallel_devices = [dev]
            self.layer_devices = [dev] * self.num_layers
            self.primary_device = dev
            self.output_device = dev
            if self.cluster_head is not None:
                self.cluster_head = self._safe_module_to(self.cluster_head, dev)
            return self

        # Start from CPU to avoid direct CUDA->CUDA parameter moves on this host.
        self.to(torch.device("cpu"))

        self.model_parallel = True
        self.parallel_devices = device_list
        self.primary_device = device_list[0]

        n_dev = len(device_list)
        self.layer_devices = [device_list[min((i * n_dev) // self.num_layers, n_dev - 1)] for i in range(self.num_layers)]
        self.output_device = self.layer_devices[-1]

        for i in range(self.num_layers):
            layer_dev = self.layer_devices[i]
            self.encoder_layers[i] = self._safe_module_to(self.encoder_layers[i], layer_dev)
            self.encoder_norms[i] = self._safe_module_to(self.encoder_norms[i], layer_dev)
            if i < self.num_layers - 1:
                self.encoder_acts[i] = self._safe_module_to(self.encoder_acts[i], layer_dev)

        self.decoder_fc1 = self._safe_module_to(self.decoder_fc1, self.output_device)
        self.decoder_fc2 = self._safe_module_to(self.decoder_fc2, self.output_device)
        self.decoder_act = self._safe_module_to(self.decoder_act, self.output_device)

        if self.mask_token_param is not None:
            self.mask_token_param.data = self._safe_tensor_to(self.mask_token_param.data, self.primary_device)

        if self.cluster_head is not None:
            self.cluster_head = self._safe_module_to(self.cluster_head, self.output_device)

        return self

    def _edge_index_for_device(self, edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
        if edge_index.device == device:
            return edge_index
        key = (device.type, device.index)
        cached = self._edge_index_cache.get(key)
        if cached is None or cached.device != device:
            cached = self._safe_tensor_to(edge_index, device)
            self._edge_index_cache[key] = cached
        return cached

    @property
    def enc_params(self):
        params = list(self.encoder_layers.parameters())
        params += list(self.encoder_norms.parameters())
        params += list(self.encoder_acts.parameters())
        if self.mask_token_param is not None:
            params.append(self.mask_token_param)
        return params

    @property
    def dec_params(self):
        params = list(self.decoder_fc1.parameters())
        params += list(self.decoder_fc2.parameters())
        params += list(self.decoder_act.parameters())
        return params

    def ensure_cluster_head(self, num_classes: int) -> nn.Linear:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")

        device = self.output_device if self.model_parallel else next(self.parameters()).device
        if self.cluster_head is None:
            self.cluster_head = nn.Linear(self.hidden_dim, int(num_classes)).to(device)
            return self.cluster_head

        if self.cluster_head.out_features == int(num_classes):
            if self.cluster_head.weight.device != device:
                self.cluster_head = self._safe_module_to(self.cluster_head, device)
            return self.cluster_head

        old = self.cluster_head
        new = nn.Linear(self.hidden_dim, int(num_classes)).to(device)
        with torch.no_grad():
            keep = min(old.out_features, int(num_classes))
            old_w = self._safe_tensor_to(old.weight, device)
            old_b = self._safe_tensor_to(old.bias, device)
            new.weight[:keep].copy_(old_w[:keep])
            new.bias[:keep].copy_(old_b[:keep])
        self.cluster_head = new
        return self.cluster_head

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        if self.cluster_head is None:
            raise RuntimeError("cluster_head is not initialized; call ensure_cluster_head first")
        if self.cluster_head.weight.device != z.device:
            self.cluster_head = self._safe_module_to(self.cluster_head, z.device)
        return self.cluster_head(z)

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(self.num_layers):
            layer_device = self.layer_devices[i] if self.layer_devices else h.device
            if h.device != layer_device:
                h = self._safe_tensor_to(h, layer_device)

            edge_index_local = self._edge_index_for_device(edge_index, layer_device)
            h = self.encoder_layers[i](h, edge_index_local)
            h = self.encoder_norms[i](h)

            if i < self.num_layers - 1:
                h = self.encoder_acts[i](h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        target_device = self.output_device if self.model_parallel else z.device
        if z.device != target_device:
            z = self._safe_tensor_to(z, target_device)

        h = self.decoder_fc1(z)
        h = self.decoder_act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.decoder_fc2(h)

    def embed(self, data: Data) -> torch.Tensor:
        if not hasattr(data, "x") or not hasattr(data, "edge_index"):
            raise ValueError("Data object must contain 'x' and 'edge_index'")

        x = data.x
        edge_index = data.edge_index

        if self.layer_devices:
            if x.device != self.primary_device:
                x = self._safe_tensor_to(x, self.primary_device)
        return self._encode(x, edge_index)

    def forward(self, data: Data, mask_info=None):
        if mask_info is not None and isinstance(mask_info, dict) and "x" in mask_info:
            x = mask_info["x"]
            if self.layer_devices and x.device != self.primary_device:
                x = self._safe_tensor_to(x, self.primary_device)
            z = self._encode(x, data.edge_index)
        else:
            z = self.embed(data)
        x_rec = self.reconstruct(z)
        return z, x_rec
