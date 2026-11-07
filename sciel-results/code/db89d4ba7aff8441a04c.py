"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Heuristic using weighted harmonic mean across constraints.
    
    For each item, compute prize/weight per constraint, then combine
    via harmonic mean weighted by constraint tightness (higher weight
    for constraints where the item is relatively expensive).
    
    The harmonic mean is a smooth surrogate for the min-ratio,
    penalizing items that are inefficient in any single constraint.
    """
    # Per-constraint efficiency: prize / weight for each constraint
    # Shape: (5, 100)
    eff = prizes / (weights + 1e-9)
    
    # Constraint tightness: constraints with higher average weight are tighter
    # (they will be the bottleneck). Weight = mean weight per constraint.
    # Shape: (5,)
    tightness = weights.mean(axis=1)  # (5,)
    
    # Weighted harmonic mean: H = 1 / sum(w_i / eff_i)
    # where w_i are normalized tightness weights
    w = tightness / (tightness.sum() + 1e-9)  # (5,)
    
    # Compute weighted harmonic mean per item
    # 1 / sum_j (w_j / eff_j[i])
    inv_eff = 1.0 / (eff + 1e-9)  # (5, 100)
    weighted_sum = (w[:, np.newaxis] * inv_eff).sum(axis=0)  # (100,)
    
    # Harmonic mean gives a score where lower values mean more attractive
    # (since it's 1/efficiency). We want higher = more attractive.
    # So we take reciprocal: harmonic mean of efficiencies
    h_mean = 1.0 / (weighted_sum + 1e-9)  # (100,)
    
    # Scale by mean prize to keep magnitude reasonable
    scale = prizes.mean()
    result = h_mean * scale
    
    # Ensure positivity and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
