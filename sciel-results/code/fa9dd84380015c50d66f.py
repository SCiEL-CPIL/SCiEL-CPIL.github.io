"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    pair_sum = d[:, None] + d[None, :]
    waste = capacity - pair_sum
    
    # Base: reward tight fits with inverse waste, sharpened by high power
    base = np.where(waste >= 0, 1.0 / (waste + 1.0), 1e-9)
    power = 10.0
    base = np.power(base, power)
    
    # Complementarity structure: sort items and identify ideal partner pairs
    # via two-pointer sweep. Assign higher attractiveness to pairs that are
    # near-optimal complements in sorted order.
    sorted_idx = np.argsort(d)
    sorted_d = d[sorted_idx]
    
    # Two-pointer sweep to find ideal complement pairs
    # For each item i (in sorted order), find j such that sorted_d[i] + sorted_d[j] <= capacity
    # and the sum is as large as possible (tightest fit)
    ideal_partner = np.zeros(n, dtype=np.int64)
    j = n - 1
    for i in range(n):
        while j >= i and sorted_d[i] + sorted_d[j] > capacity:
            j -= 1
        if j >= i:
            ideal_partner[i] = j
        else:
            ideal_partner[i] = i  # fallback: pair with self
    
    # Map back to original indices
    # For each original item, its ideal partner in original index space
    ideal_partner_orig = np.zeros(n, dtype=np.int64)
    for k in range(n):
        orig_idx = sorted_idx[k]
        partner_sorted_idx = ideal_partner[k]
        partner_orig_idx = sorted_idx[partner_sorted_idx]
        ideal_partner_orig[orig_idx] = partner_orig_idx
    
    # Build complementarity score: high when j is the ideal partner of i
    # Use a smooth kernel around the ideal partner
    # For each i, score[i, j] is high if j == ideal_partner_orig[i]
    complementarity = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ideal_j = ideal_partner_orig[i]
        # Gaussian-like kernel centered at ideal_j
        # Since items are discrete, use a simple distance-based score
        for j in range(n):
            if d[i] + d[j] <= capacity:
                # Score based on how close j is to the ideal partner
                # Use the sorted position distance
                pos_i = np.where(sorted_idx == i)[0][0]
                pos_j = np.where(sorted_idx == j)[0][0]
                pos_ideal = np.where(sorted_idx == ideal_j)[0][0]
                dist = abs(pos_j - pos_ideal)
                # Exponential decay with distance
                complementarity[i, j] = np.exp(-0.5 * dist)
    
    # Combine base and complementarity
    result = base * (1.0 + 0.5 * complementarity)
    
    # Ensure non-finite values are replaced
    result = np.where(np.isfinite(result), result, 1e-9)
    
    # Ensure all non-positive become 1e-9
    result = np.where(result > 0, result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
