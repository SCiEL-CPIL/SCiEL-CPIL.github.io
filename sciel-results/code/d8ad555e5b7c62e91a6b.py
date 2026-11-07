"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Heuristic using geometric blending of min-ratio (bottleneck) and
    weighted harmonic mean (overall efficiency) across constraints.
    
    The min-ratio captures the worst-case constraint efficiency (bottleneck),
    while the harmonic mean captures overall multi-constraint efficiency.
    Geometric blending (power 1.5 for min, 0.5 for harmonic) better identifies
    items that are efficient in the tightest constraint without being overly
    penalized for moderate inefficiency in others.
    """
    n_constraints, n_items = prizes.shape
    
    # Per-constraint efficiency: prize / weight for each constraint
    # Shape: (5, 100)
    eff = prizes / (weights + 1e-9)
    
    # Min-ratio: the bottleneck efficiency for each item
    # Shape: (100,)
    min_eff = eff.min(axis=0)
    
    # Constraint tightness: constraints with higher average weight are tighter
    # (they will be the bottleneck). Weight = mean weight per constraint.
    # Shape: (5,)
    tightness = weights.mean(axis=1)  # (5,)
    
    # Normalize tightness weights
    w = tightness / (tightness.sum() + 1e-9)  # (5,)
    
    # Weighted harmonic mean: H = 1 / sum(w_i / eff_i)
    # inv_eff shape: (5, 100)
    inv_eff = 1.0 / (eff + 1e-9)
    weighted_sum = (w[:, np.newaxis] * inv_eff).sum(axis=0)  # (100,)
    h_mean = 1.0 / (weighted_sum + 1e-9)  # (100,)
    
    # Geometric blending: min_eff^1.5 * h_mean^0.5
    # This emphasizes the bottleneck while still rewarding overall efficiency
    result = (min_eff ** 1.5) * (h_mean ** 0.5)
    
    # Scale by mean prize to keep magnitude reasonable
    scale = prizes.mean()
    result = result * scale
    
    # Ensure positivity and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
