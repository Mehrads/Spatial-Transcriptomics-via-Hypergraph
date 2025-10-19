import torch
import torch.nn as nn

_ACTIVATIONS = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'prelu': nn.PReLU,
    'elu': nn.ELU,
    'leakyrelu': lambda: nn.LeakyReLU(0.2),
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    None: nn.Identity,
}

def create_activation(name):
    """Return a torch.nn module for the requested activation.

    Falls back to identity if the name is None or unrecognised.
    """
    if name is None:
        return nn.Identity()
    key = str(name).lower()
    act = _ACTIVATIONS.get(key)
    if act is None:
        return nn.Identity()
    return act() if callable(act) else act()

class NormLayer(nn.Module):
    """Simple wrapper that applies a normalisation layer when provided."""

    def __init__(self, norm_type: str, num_features: int):
        super().__init__()
        factory = create_norm(norm_type)
        self.norm = factory(num_features) if factory is not None else nn.Identity()

    def forward(self, x):
        return self.norm(x)

_DEF_NORMS = {
    None: lambda _: nn.Identity(),
    'none': lambda _: nn.Identity(),
    'batchnorm': nn.BatchNorm1d,
    'layernorm': nn.LayerNorm,
    'groupnorm': lambda num_features: nn.GroupNorm(1, num_features),
}

def create_norm(norm_type):
    """Return a callable that builds the requested norm layer."""
    if norm_type is None:
        return _DEF_NORMS[None]
    key = str(norm_type).lower()
    return _DEF_NORMS.get(key, _DEF_NORMS[None])
