import numpy as np
import time

def heuristics(demands, capacity):
    """
    Heuristic matrix for ACO bin packing: attractiveness of placing item j after item i.
    
    Uses a refined product-to-waste-squared ratio with balance and tightness bonuses.
    The heuristic rewards:
    - Tight packing (low waste, squared penalty)
    - Large items (high product)
    - Balanced pairs (both items contribute significantly)
    - Perfect fits (waste = 0 get extra boost)
    """
    start_time = time.time()

    d = np.asarray(demands, dtype=np.float64)
    n = len(d)

    if n == 0:
        return np.empty((0, 0))

    # Pairwise sums and products
    sums = d[:, None] + d[None, :]
    products = d[:, None] * d[None, :]

    # Feasibility mask
    fits = sums <= capacity

    # Waste = capacity - sum (non-negative where fits)
    waste = np.where(fits, capacity - sums, 1.0)

    # Smooth denominator: (waste + 1.0)^2
    # For waste=0 (perfect fit): denom = 1.0
    # For waste=1: denom = 4.0
    # For waste=5: denom = 36.0
    # This strongly rewards tight fits
    denom = (waste + 1.0) ** 2

    # Balance factor: prefer pairs where both items are reasonably large
    # min(d_i, d_j) / (d_i + d_j) ranges from 0.5 (one dominates) to 1.0 (equal)
    min_items = np.minimum(d[:, None], d[None, :])
    safe_sums = np.where(sums > 0, sums, 1.0)
    balance = np.where(fits, min_items / safe_sums, 0.5)
    
    # Balance boost: stronger weight to prefer balanced pairs
    # Range: 1.0 (unbalanced) to 1.5 (perfectly balanced)
    balance_boost = 1.0 + 1.0 * (balance - 0.5)

    # Tightness bonus: extra boost for very tight fits (waste <= 2)
    # This creates a stronger preference for near-perfect fits
    tightness_bonus = np.where(fits & (waste <= 2.0), 1.5, 1.0)

    # Score: product * balance_boost * tightness_bonus / (waste + 1)^2
    raw_scores = products * balance_boost * tightness_bonus / denom

    # Mask non-fitting pairs to 1e-9
    H = np.where(fits, raw_scores, 1e-9)

    # Sanitize: replace any non-finite values
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)

    # Shape check
    if H.shape != (n, n):
        raise ValueError("Wrong shape")

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H

    return H
