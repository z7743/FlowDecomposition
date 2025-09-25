# flow_decomposition/utils/metrics.py
from __future__ import annotations
from typing import Callable, Dict, Optional
import torch

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# All metrics return [batch, 1, comp, comp] given A,B with shape [batch, samples, dim, comp, comp]

def _double_center(D: torch.Tensor) -> torch.Tensor:
    mu_r = D.mean(dim=-1, keepdim=True)
    mu_c = D.mean(dim=-2, keepdim=True)
    mu_a = D.mean(dim=(-1, -2), keepdim=True)
    return D - mu_r - mu_c + mu_a

def _flatten_AB(A: torch.Tensor, B: torch.Tensor):
    # A,B: [B, S, D, C, C] -> A2,B2: [B*C*C, S, D], C = comp
    Bch, S, D, C, _ = A.shape
    A2 = A.permute(0, 3, 4, 1, 2).reshape(Bch * C * C, S, D)
    B2 = B.permute(0, 3, 4, 1, 2).reshape(Bch * C * C, S, D)
    return A2, B2, Bch, C

def batch_corr(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    muA = A.mean(dim=1, keepdim=True)
    muB = B.mean(dim=1, keepdim=True)
    num = ((A - muA) * (B - muB)).sum(dim=1)
    den = torch.sqrt(((A - muA).pow(2)).sum(dim=1) * ((B - muB).pow(2)).sum(dim=1) + eps)
    r = num / den
    Bch, _, _, C, _ = A.shape
    return r.mean(dim=1, keepdim=True)

def batch_abs_corr(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    muA = A.mean(dim=1, keepdim=True)
    muB = B.mean(dim=1, keepdim=True)
    num = (A - muA).abs().mul((B - muB).abs()).sum(dim=1)
    den = torch.sqrt(((A - muA).abs().pow(2)).sum(dim=1) * ((B - muB).abs().pow(2)).sum(dim=1) + eps)
    r = num / den
    Bch, _, _, C, _ = A.shape
    return r.mean(dim=1, keepdim=True)

def batch_ccc(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12, unbiased: bool = True) -> torch.Tensor:
    # Lin's CCC on flattened observations (samples*dim)
    A2 = A.flatten(start_dim=1, end_dim=2)  # [B, N, C, C]
    B2 = B.flatten(start_dim=1, end_dim=2)
    N  = A2.shape[1]
    muA, muB = A2.mean(1, True), B2.mean(1, True)
    if unbiased and N > 1:
        varA = ((A2 - muA).pow(2)).sum(1, True) / (N - 1)
        varB = ((B2 - muB).pow(2)).sum(1, True) / (N - 1)
        cov  = (((A2 - muA) * (B2 - muB)).sum(1, True)) / (N - 1)
    else:
        varA = ((A2 - muA).pow(2)).mean(1, True)
        varB = ((B2 - muB).pow(2)).mean(1, True)
        cov  = ((A2 - muA) * (B2 - muB)).mean(1, True)
    ccc = (2.0 * cov) / (varA + varB + (muA - muB).pow(2) + eps)
    return ccc  # already [B,1,C,C]

def batch_dccc(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8, signed: bool = False) -> torch.Tensor:
    # Distance-CCC on flattened observations
    A2, B2, Bch, C = _flatten_AB(A, B)
    # means across observations
    muA, muB = A2.mean(1), B2.mean(1)
    mean_diff_sq = (muA - muB).pow(2).sum(1)
    # pairwise dists and double-centering
    DA = torch.cdist(A2, A2, p=2, compute_mode="use_mm_for_euclid_dist")
    DB = torch.cdist(B2, B2, p=2, compute_mode="use_mm_for_euclid_dist")
    A_dc, B_dc = _double_center(DA), _double_center(DB)
    dCov   = (A_dc * B_dc).mean(dim=(-1, -2))
    dVar_A = (A_dc.pow(2)).mean(dim=(-1, -2))
    dVar_B = (B_dc.pow(2)).mean(dim=(-1, -2))
    denom = dVar_A + dVar_B + mean_diff_sq + eps
    dccc = (2.0 * dCov) / denom
    if signed:
        A0 = A2 - muA.unsqueeze(1)
        B0 = B2 - muB.unsqueeze(1)
        num = (A0 * B0).mean(dim=(1, 2))
        den = torch.sqrt(A0.pow(2).mean(dim=(1, 2)) * B0.pow(2).mean(dim=(1, 2)) + eps)
        dccc = dccc * torch.sign(num / den)
    # degenerate identical-constant case -> 1
    deg = (dVar_A + dVar_B + mean_diff_sq) < eps
    if torch.any(deg):
        dccc = dccc.clone()
        dccc[deg] = torch.tensor(1.0, dtype=dccc.dtype, device=dccc.device)
    return dccc.reshape(Bch, 1, C, C)

def batch_dcor(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    A2, B2, Bch, C = _flatten_AB(A, B)
    DA = torch.cdist(A2, A2, p=2, compute_mode="use_mm_for_euclid_dist")
    DB = torch.cdist(B2, B2, p=2, compute_mode="use_mm_for_euclid_dist")
    A_dc, B_dc = _double_center(DA), _double_center(DB)
    dCov   = (A_dc * B_dc).mean(dim=(-1, -2))
    dVar_A = (A_dc.pow(2)).mean(dim=(-1, -2))
    dVar_B = (B_dc.pow(2)).mean(dim=(-1, -2))
    dCor = dCov / (torch.sqrt(dVar_A * dVar_B) + eps)
    return dCor.reshape(Bch, 1, C, C)

def batch_hsic_rbf(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8, normalized: bool = False) -> torch.Tensor:
    # RBF-HSIC across samples; optional normalization -> CKA
    A64 = A.to(torch.float64)
    B64 = B.to(torch.float64)
    A2 = A64.permute(0, 3, 4, 1, 2).reshape(-1, A64.shape[1], A64.shape[2])
    B2 = B64.permute(0, 3, 4, 1, 2).reshape(-1, B64.shape[1], B64.shape[2])

    def rbf_gram(X):
        D2 = torch.cdist(X, X, p=2, compute_mode="use_mm_for_euclid_dist").pow(2)
        sigma2 = (D2.mean(dim=(-1, -2), keepdim=True) + eps).detach()
        return torch.exp(-D2 / (2.0 * sigma2))

    def center(K):
        mu_r = K.mean(dim=-1, keepdim=True)
        mu_c = K.mean(dim=-2, keepdim=True)
        mu_a = K.mean(dim=(-1, -2), keepdim=True)
        return K - mu_r - mu_c + mu_a

    K = center(rbf_gram(A2))
    L = center(rbf_gram(B2))
    hsic = (K * L).mean(dim=(-1, -2))
    if normalized:
        kk = (K * K).mean(dim=(-1, -2)) + eps
        ll = (L * L).mean(dim=(-1, -2)) + eps
        hsic = hsic / torch.sqrt(kk * ll)
    batch, comp = A.shape[0], A.shape[-1]
    out = hsic.reshape(batch, comp, comp).unsqueeze(1)
    return out.to(dtype=A.dtype)

def batch_cka_rbf(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return batch_hsic_rbf(A, B, eps=eps, normalized=True)


def batch_dcor_weighted(
    A: torch.Tensor, a: torch.Tensor,
    B: torch.Tensor, b: torch.Tensor,
    theta: float = 1.0,
    eps: float = 1e-8,
    subtract_unweighted: bool = True,
) -> torch.Tensor:
    # A,a,B,b with shapes matching your implementation:
    # A: [B, subset, Dx, C, C], a: [B, sample, Dx, C, C]
    # B: [B, subset, Dy, C, C], b: [B, sample, Dy, C, C]
    Bch, S, Dx, C, _ = a.shape
    _,  L, _,  _, _ = A.shape

    def _flat(x):  # -> [B*C*C, n, D]
        return x.permute(0, 3, 4, 1, 2).reshape(Bch * C * C, -1, x.shape[2])

    a_f, A_f, b_f, B_f = map(_flat, (a, A, b, B))

    pair = lambda X, Y: torch.cdist(X, Y, p=2, compute_mode="use_mm_for_euclid_dist")
    DA = pair(a_f, A_f)   # [B*C*C, S, L]
    DB = pair(b_f, B_f)   # [B*C*C, S, L]

    # Local weights from Y-space distances
    avg = DB.mean(dim=(-1, -2), keepdim=True)
    Wloc = torch.exp(-theta * DB / (avg + eps))

    def _dc(W, Xdc, Ydc):
        wsum  = W.sum(dim=(-1, -2)) + eps
        dCov  = (W * Xdc * Ydc).sum(dim=(-1, -2)) / wsum
        dVarX = (W * Xdc.pow(2)).sum(dim=(-1, -2)) / wsum
        dVarY = (W * Ydc.pow(2)).sum(dim=(-1, -2)) / wsum
        return dCov / (torch.sqrt(dVarX * dVarY) + eps)

    # double-center each sample-weighted distance block
    def _dcen(D):
        mr = D.mean(dim=-1, keepdim=True)
        mc = D.mean(dim=-2, keepdim=True)
        ma = D.mean(dim=(-1, -2), keepdim=True)
        return D - mr - mc + ma

    A_dc = _dcen(DA)
    B_dc = _dcen(DB)
    dCor_w = _dc(Wloc, A_dc, B_dc)

    if subtract_unweighted:
        dCor_u = _dc(torch.ones_like(Wloc), A_dc, B_dc)
        dCor_u = dCor_u.reshape(Bch, C, C).unsqueeze(1)
        dCor_w = dCor_w.reshape(Bch, C, C).unsqueeze(1)
        eye = torch.eye(C, device=dCor_w.device).view(1, 1, C, C)
        out = dCor_w - dCor_u * eye
        return out
    else:
        return dCor_w.reshape(Bch, C, C).unsqueeze(1)
    

def batch_neg_nrmse(A: torch.Tensor, B: torch.Tensor, T: float = 0.5, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalized negative RMSE over (samples, dim) for each [comp, comp] entry.
    neg_nrmse = - RMSE(A,B) / RMSE(mean(B), B)
    Returns: [B, 1, C, C]
    """
    # RMSE(A, B)
    mse = (A - B).pow(2).mean(dim=(1, 2), keepdim=False)          # [B, C, C]
    rmse = torch.sqrt(mse + eps)                                   # [B, C, C]

    # Baseline RMSE(mean(B), B) == std of B over (samples, dim)
    muB = B.mean(dim=(1, 2), keepdim=True)                         # [B, 1, 1, C, C]
    varB = (B - muB).pow(2).mean(dim=(1, 2), keepdim=False)        # [B, C, C]
    rmse_base = torch.sqrt(varB + eps)                             # [B, C, C]

    #neg_nrmse = torch.exp(- ((rmse / (rmse_base + eps))**2)/2)                         # [B, C, C]

    neg_nrmse = torch.exp(- ((1/T) * torch.pow(rmse / (rmse_base + eps), 2)))   
    return neg_nrmse.unsqueeze(1).to(dtype=A.dtype)                # [B, 1, C, C]

# ---- Registry ----
_METRICS: Dict[str, Metric] = {
    "corr":    batch_corr,
    "abs_corr": batch_abs_corr,
    "ccc":     batch_ccc,
    "dccc":    batch_dccc,
    "dcorr":   batch_dcor,
    "hsic":    batch_hsic_rbf,
    "cka":     batch_cka_rbf,
    "neg_nrmse": batch_neg_nrmse, 
}

def get_metric(name: str) -> Metric:
    if name not in _METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {list(_METRICS)}")
    return _METRICS[name]