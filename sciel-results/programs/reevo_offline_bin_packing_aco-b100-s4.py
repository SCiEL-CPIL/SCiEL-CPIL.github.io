import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def heuristics(demands, capacity):
    """
    Heuristic matrix for ACO bin packing: attractiveness of placing item j after item i.
    
    Uses a maximum weight matching approach with smooth complementarity scoring:
    1. Compute smooth complementarity scores (soft-min of waste) for all feasible pairs
    2. Find optimal disjoint pairs via maximum weight matching
    3. Boost matched pairs with a smooth, tightness-dependent factor
    4. Apply inverse-gap penalty to strongly discourage wasteful pairings
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

    # Smooth complementarity: use soft-min to smoothly penalize waste
    # Higher score when waste is small (tight fit)
    # Use log-sum-exp style smooth min: soft_min(x, y) ≈ -log(exp(-x/t) + exp(-y/t))
    # Here we want a smooth function that is high when waste is low
    # Complementarity = 1 / (1 + waste^alpha) with smooth alpha
    alpha = 2.0
    complementarity = 1.0 / (1.0 + waste ** alpha)

    # Base score: product scaled by complementarity
    base_scores = products * complementarity

    # Mask non-fitting pairs
    base_scores = np.where(fits, base_scores, 0.0)

    # Sanitize base scores
    if not np.all(np.isfinite(base_scores)):
        base_scores = np.where(np.isfinite(base_scores), base_scores, 0.0)

    # Normalize base scores to [0, 1] range for stable matching
    max_base = base_scores.max()
    if max_base > 0:
        base_scores_norm = base_scores / max_base
    else:
        base_scores_norm = base_scores

    # Find maximum weight matching of disjoint pairs
    matching_weights = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if fits[i, j]:
                matching_weights[i, j] = base_scores_norm[i, j]
                matching_weights[j, i] = base_scores_norm[i, j]

    # Find optimal assignment (minimizes negative = maximizes positive)
    row_ind, col_ind = linear_sum_assignment(-matching_weights)

    # Build set of matched pairs with their weights
    matched_pairs = {}
    for r, c in zip(row_ind, col_ind):
        if r != c and matching_weights[r, c] > 0:
            matched_pairs[(r, c)] = matching_weights[r, c]
            matched_pairs[(c, r)] = matching_weights[r, c]

    # Construct final heuristic with smooth, tightness-dependent boost
    # Boost factor depends on how tight the pair is: tighter pairs get higher boost
    # Use smooth interpolation: boost = 1 + max_boost * tightness^beta
    max_boost = 8.0
    beta = 1.5
    H = base_scores.copy()
    for (i, j), w in matched_pairs.items():
        # Tightness is w (normalized base score, higher = tighter)
        boost_factor = 1.0 + max_boost * (w ** beta)
        H[i, j] = base_scores[i, j] * boost_factor

    # Apply inverse-gap penalty: strongly penalize pairs that leave lots of space
    # This encourages the ACO to prefer pairs that nearly fill the bin
    # Penalty: multiply by (1 + waste / capacity)^(-gamma) where gamma > 1
    gamma = 3.0
    gap_penalty = (1.0 + waste / capacity) ** (-gamma)
    H = H * gap_penalty

    # Ensure non-fitting pairs are very low
    H = np.where(fits, H, 1e-9)

    # Sanitize: replace any non-finite values
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)

    # Ensure all values are positive (as per task requirements)
    H = np.maximum(H, 1e-9)

    # Shape check
    if H.shape != (n, n):
        raise ValueError("Wrong shape")

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H

    return H
