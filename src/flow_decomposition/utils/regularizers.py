# flow_decomposition/utils/regularizers.py
import torch
from typing import Callable, Dict, Iterable


def h_norm(params) -> torch.Tensor:
    params = [p for p in params if p.requires_grad]
    if not params:
        return torch.tensor(0.)
    p0 = params[0]
    l1 = sum(p.abs().sum() for p in params)
    l2 = sum(p.norm(2)     for p in params)
    n  = torch.tensor(sum(p.numel() for p in params), dtype=p0.dtype, device=p0.device)
    return 1 - (torch.sqrt(n) - (l1 / (l2 + 1e-12))) / (torch.sqrt(n) - 1)

def gini(params, dtype=None, device=None) -> torch.Tensor:
    w = torch.cat([p.view(-1).abs() for p in params if p.requires_grad])
    if w.numel() == 0:
        return torch.tensor(0., dtype=dtype or torch.float32, device=device)
    w_sorted, _ = torch.sort(w)
    n = w_sorted.numel()
    idx = torch.arange(1, n + 1, device=w_sorted.device, dtype=w_sorted.dtype)
    g = (2.0 * (idx * w_sorted).sum()) / (n * w_sorted.sum() + 1e-12) - (n + 1) / n
    return 1.0 - g

def l1_over_l2(params) -> torch.Tensor:
    l1 = sum(p.abs().sum() for p in params if p.requires_grad)
    l2 = sum(p.norm(2)     for p in params if p.requires_grad)
    return l1 / (l2 + 1e-12)

Regularizer = Callable[[Iterable[torch.nn.Parameter]], torch.Tensor]

_REGULARIZERS: Dict[str, Regularizer] = {
    "h_norm": h_norm,
    "gini": gini,
    "l1_l2": l1_over_l2,
}

def get_regularizer(name: str) -> Regularizer:
    try:
        return _REGULARIZERS[name]
    except KeyError:
        raise ValueError(f"Unknown regularizer: {name}. Available: {list(_REGULARIZERS.keys())}")

def list_regularizers():
    return list(_REGULARIZERS.keys())