import numpy as np
import time

def heuristics(prizes, weights):
    # Deadline guard: we must return well before 330s wall-clock
    deadline = time.time() + 30.0  # be very conservative, this function should be fast
    
    # Handle shape
    if prizes.ndim == 2:
        P = prizes.sum(axis=0)  # (100,)
    else:
        P = np.asarray(prizes).flatten()
        
    if weights.ndim == 2:
        W = weights
    else:
        W = weights.reshape(1, -1)
        
    n_constraints, n_items = W.shape
    
    # Per-constraint efficiency: for each constraint c and item i
    # eff[c, i] = P[i] / W[c, i]
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = P[None, :] / (W + 1e-12)
    eff = np.where(np.isfinite(eff), eff, 0.0)
    
    # Scarcity estimation: for each constraint, look at the distribution of efficiencies
    # Constraints where the top items are much more efficient than average are "scarcer"
    # because those few items dominate that resource's usage
    
    # Top-k efficiency per constraint (k = top 20%)
    k = max(1, n_items // 5)
    top_k_eff = np.zeros(n_constraints)
    mean_eff = np.zeros(n_constraints)
    for c in range(n_constraints):
        top_k_eff[c] = np.partition(eff[c], -k)[-k].mean()
        mean_eff[c] = eff[c].mean()
    
    # Scarcity score: high top-k/mean ratio indicates a constraint is "tight"
    # because a few items are much better at using that resource
    with np.errstate(divide='ignore', invalid='ignore'):
        scarcity_ratio = top_k_eff / (mean_eff + 1e-12)
    scarcity_ratio = np.where(np.isfinite(scarcity_ratio), scarcity_ratio, 1.0)
    
    # Normalize scarcity weights to sum to 1
    scarcity = scarcity_ratio / (scarcity_ratio.sum() + 1e-12)
    
    # Marginal utility: weighted sum of per-constraint efficiencies
    marginal_util = (scarcity[:, None] * eff).sum(axis=0)
    
    # Base efficiency: total prize / total weight
    W_sum = W.sum(axis=0)
    base_eff = P / (W_sum + 1e-12)
    base_eff = np.where(np.isfinite(base_eff), base_eff, 0.0)
    
    # Soft-max bottleneck penalty: penalize items that are heavy in any single constraint
    # Using log-sum-exp for smoothness
    beta = 3.0
    W_max = np.max(W, axis=0)
    log_sum_exp = W_max + np.log(np.sum(np.exp((W - W_max) * beta), axis=0)) / beta
    soft_bottleneck = log_sum_exp
    
    # Constraint-specific scarcity premium:
    # For each item, identify the most binding constraint (largest fraction of total weight)
    # and scale by inverse remaining capacity fraction
    # Total weight per item
    W_total = W_sum + 1e-12
    # Fraction of total weight in each constraint: frac[c, i] = W[c, i] / W_total[i]
    frac = W / W_total[None, :]
    # Most binding constraint index per item
    most_binding = np.argmax(frac, axis=0)  # (100,)
    
    # Remaining capacity fraction for the most binding constraint
    # Capacity is 25 for each constraint
    capacity = 25.0
    # The "remaining capacity fraction" is roughly capacity / (sum of weights of items that use this constraint)
    # But since we don't know which items are selected, we approximate:
    # Use the total weight in that constraint across all items as a proxy for how "full" it is
    # Actually, let's use a simpler heuristic: the fraction itself tells us how much of the item's
    # weight is in the most binding constraint. Higher fraction means more constrained.
    # So we penalize items with high fraction in their most binding constraint.
    
    # Get the fraction for the most binding constraint per item
    binding_frac = frac[most_binding, np.arange(n_items)]
    
    # Scarcity premium: scale by 1 / (1 + binding_frac * scarcity_factor)
    # Items with high binding fraction in a scarce constraint get less premium
    scarcity_factor = 2.0
    scarcity_premium = 1.0 / (1.0 + binding_frac * scarcity_factor)
    
    # Combine: marginal utility * base efficiency / soft_bottleneck * scarcity_premium
    H = marginal_util * base_eff / (soft_bottleneck + 1e-12) * scarcity_premium
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H), H, 1e-9)
    H = np.maximum(H, 1e-9)
    
    # Deadline guard
    if time.time() > deadline - 5.0:
        return H
    
    return H
