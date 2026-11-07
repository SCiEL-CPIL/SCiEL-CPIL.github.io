import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 320.0  # safety margin before 330s kill


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Combines scarcity-aware marginal utility with bottleneck efficiency.
    1. Scarcity weighting: constraints with higher top-k efficiency are "scarcer"
       because the best items use them most efficiently, meaning they bind first.
    2. Marginal utility: weighted sum of per-constraint efficiencies using scarcity weights.
    3. Bottleneck efficiency: geometric mean of per-constraint prize/weight ratios,
       weighted by scarcity to emphasize tighter constraints.
    4. Final score: product of marginal utility and weighted geometric mean,
       normalized and blended with aggregate efficiency.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    eps = 1e-9
    capacity = 25.0
    
    if prizes.ndim == 2:
        P = prizes.sum(axis=0)
    else:
        P = prizes
        
    if weights.ndim == 2:
        W = weights
    else:
        W = weights.reshape(1, -1)
        
    n_constraints, n_items = W.shape
    
    # Per-constraint efficiency: P[i] / W[c, i]
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = P[None, :] / (W + eps)
    eff = np.where(np.isfinite(eff), eff, 0.0)
    
    # Scarcity weight based on top-k efficiency per constraint
    # Constraints with higher top-k efficiency are scarcer (best items bind them first)
    k = max(1, n_items // 5)  # top 20% of items
    top_k_eff = np.zeros(n_constraints)
    for c in range(n_constraints):
        top_k_eff[c] = np.partition(eff[c], -k)[-k].mean()
    
    # Also consider remaining capacity scarcity
    total_weight_per_constraint = np.sum(W, axis=1)  # (5,)
    remaining_capacity = np.maximum(capacity - total_weight_per_constraint, eps)
    remaining_frac = remaining_capacity / capacity  # (5,)
    
    # Combine scarcity signals: top-k efficiency and remaining capacity
    # Higher top_k_eff means more scarce; lower remaining_frac means more scarce
    scarcity_topk = top_k_eff / (top_k_eff.sum() + eps)
    scarcity_remaining = (1.0 / (remaining_frac + eps))
    scarcity_remaining = scarcity_remaining / (scarcity_remaining.sum() + eps)
    
    # Blend scarcity signals
    alpha_scarcity = 0.6
    scarcity = alpha_scarcity * scarcity_topk + (1 - alpha_scarcity) * scarcity_remaining
    
    # Marginal utility: weighted sum of per-constraint efficiencies
    marginal_util = (scarcity[:, None] * eff).sum(axis=0)  # (100,)
    
    # Weighted geometric mean of efficiencies (bottleneck-aware)
    # Use log-space for stability
    log_eff = np.log(eff + eps)  # (5, 100)
    weighted_log = (scarcity[:, None] * log_eff).sum(axis=0)  # (100,)
    weighted_geo = np.exp(weighted_log)  # (100,)
    
    # Aggregate efficiency as baseline
    W_sum = W.sum(axis=0)
    base_eff = P / (W_sum + eps)
    
    # Combine: marginal utility * weighted geometric mean, scaled by base efficiency
    # This rewards items that are efficient in scarce constraints AND generally efficient
    H_core = marginal_util * weighted_geo * base_eff
    
    # Normalize to mean 1 for stable combination
    H_core_mean = H_core.mean()
    if H_core_mean > 0:
        H_core_norm = H_core / H_core_mean
    else:
        H_core_norm = np.ones_like(H_core)
    
    base_eff_mean = base_eff.mean()
    if base_eff_mean > 0:
        base_eff_norm = base_eff / base_eff_mean
    else:
        base_eff_norm = np.ones_like(base_eff)
    
    # Blend: weight the scarcity-aware core more heavily
    alpha_blend = 0.7
    H = alpha_blend * H_core_norm + (1 - alpha_blend) * base_eff_norm
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    H = np.maximum(H, 1e-9)
    
    # Deadline guard: if we're close to the limit, return immediately
    if time.time() > _DEADLINE:
        return H
    
    return H
