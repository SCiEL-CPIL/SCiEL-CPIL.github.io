"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the ant colony heuristic.
    
    Strategy: Competitive exclusion score blended with harmonic-mean ratio.
    
    For each item, we estimate its marginal utility relative to the opportunity cost
    of consuming constraint capacity. For each constraint c, we compute:
      - The item's prize per unit of that constraint: p[c,i] / w[c,i]
      - The "competition" from other items: sum of prizes of items that would
        also be attractive in constraint c, weighted by their weight share.
    
    The competitive exclusion score captures: an item is attractive if its prize
    is high relative to the total prize mass competing for the same constraint
    resources. This directly models opportunity cost in a near-full knapsack.
    
    We blend this with the harmonic mean of per-constraint ratios (which acts as
    a smooth surrogate for the min-ratio bottleneck) to get a robust heuristic.
    """
    # Handle input shapes: prizes and weights should be (5, 100)
    if prizes.ndim == 2:
        p = prizes  # (5, 100)
    else:
        # If prizes is 1D (100,), broadcast to (5, 100)
        p = np.tile(prizes, (weights.shape[0], 1))
    
    w = weights  # (5, 100)
    eps = 1e-9
    n_constraints = p.shape[0]  # 5
    n_items = p.shape[1]  # 100
    
    # --- Component 1: Harmonic mean of per-constraint ratios ---
    ratios = p / (w + eps)  # (5, 100)
    inv_ratios = 1.0 / (ratios + eps)  # (5, 100)
    mean_inv = np.mean(inv_ratios, axis=0)  # (100,)
    h_harmonic = 1.0 / (mean_inv + eps)  # (100,)
    
    # --- Component 2: Competitive exclusion score ---
    # For each constraint c, the "opportunity cost" of item i is proportional to
    # the total prize of items competing for the same constraint resource.
    # 
    # Intuition: If many items have high prize in constraint c, then selecting
    # any single item in constraint c has lower marginal value because it
    # displaces other high-prize items.
    #
    # For constraint c, item i's competitive score:
    #   comp_c(i) = p[c,i] / (sum_j p[c,j] * w[c,j] / w[c,i])
    #             = p[c,i] * w[c,i] / sum_j(p[c,j] * w[c,j])
    #
    # This is: prize * weight / total_prize_weight for that constraint.
    # It measures: how much of the constraint's prize mass does this item represent,
    # weighted by how much capacity it consumes.
    #
    # Alternative: use prize relative to sum of all prizes in that constraint:
    #   comp_c(i) = p[c,i] / sum_j(p[c,j])
    # This is simpler: what fraction of the total prize in constraint c does item i hold?
    # Items with high share are more attractive because they dominate that constraint.
    
    # Total prize per constraint: (5,)
    total_prize_per_constraint = np.sum(p, axis=1)  # (5,)
    
    # Share of prize per constraint: (5, 100)
    prize_share = p / (total_prize_per_constraint[:, np.newaxis] + eps)  # (5, 100)
    
    # Weighted by weight: items that use more capacity but have high prize share
    # are more valuable. Use p/w ratio modulated by prize share.
    # 
    # Let's try: for each constraint, score = (p / w) * prize_share
    # This rewards items that are both efficient (high p/w) AND dominant (high share).
    weighted_share = ratios * prize_share  # (5, 100)
    
    # Geometric mean across constraints for competitive score
    # log-geometric mean = mean(log(x))
    log_ws = np.log(weighted_share + eps)  # (5, 100)
    geo_mean_log = np.mean(log_ws, axis=0)  # (100,)
    h_competitive = np.exp(geo_mean_log)  # (100,)
    
    # --- Blend the two components ---
    # Normalize both to [0, 1] range
    def normalize_positive(arr):
        arr = np.where(arr > 0, arr, eps)
        mn = np.min(arr)
        mx = np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.ones_like(arr)
    
    h1 = normalize_positive(h_harmonic)  # (100,)
    h2 = normalize_positive(h_competitive)  # (100,)
    
    # Blend: 50/50 initially, could tune
    alpha = 0.5
    h = alpha * h1 + (1 - alpha) * h2
    
    # Ensure positive values as per contract
    h = np.where(h > 0, h, 1e-9)
    
    # Scale to reasonable range
    max_h = np.max(h)
    if max_h > 0:
        h = h / max_h
    
    # Ensure minimum value
    h = np.maximum(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
