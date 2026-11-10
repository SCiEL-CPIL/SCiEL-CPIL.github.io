"""Heuristic design for ant colony bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Return an n x n matrix of attractiveness for placing item j right after item i.
    Larger values mean more attractive.
    
    Design: We want to encourage pairs (i, j) that together fill a bin tightly.
    The ideal is demands[i] + demands[j] == capacity (perfect fit).
    We use a smooth surrogate that peaks at exact fit and decays smoothly otherwise.
    
    For pairs that don't fit (sum > capacity), we set a very low value (1e-9).
    For pairs that fit, we use a function that rewards tightness:
        h(i,j) = exp(-alpha * |demands[i] + demands[j] - capacity|^2 / capacity^2)
    This gives a Gaussian bump centered at the perfect fit.
    
    Additionally, we can add a small term that prefers larger items to be placed
    together (since they consume more space and are harder to place), but the
    primary signal is the tightness of the pair sum.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Compute all pair sums
    pair_sum = d[:, None] + d[None, :]
    
    # Mask: only pairs that fit
    fits = pair_sum <= capacity
    
    # For fitting pairs, compute tightness score
    # Perfect fit is when pair_sum == capacity
    # Use a Gaussian decay around the perfect fit
    # Scale: capacity is 150, demands are 20-99, so pair sums range from 40 to 198
    # The range of fitting pair sums is [40, 150]
    # We want the heuristic to be highest at pair_sum = capacity (150)
    
    # Use: h = exp(-((pair_sum - capacity)^2) / (2 * sigma^2))
    # Choose sigma to control the width. A smaller sigma makes it more peaked.
    # With sigma = capacity/10 = 15, the Gaussian decays to ~0.6 at +/-15 from center.
    # This seems reasonable.
    
    sigma = capacity / 10.0
    diff = pair_sum - capacity
    gaussian = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # For pairs that don't fit, set to 1e-9
    h = np.where(fits, gaussian, 1e-9)
    
    # Ensure all values are positive and finite
    h = np.where(h > 0, h, 1e-9)
    h = np.where(np.isfinite(h), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
