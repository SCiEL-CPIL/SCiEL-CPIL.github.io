"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    """
    Computes a heuristic matrix for the ant colony optimizer.
    
    Strategy:
    Combine two complementary signals:
    1. Tight packing reward: 1/(remaining+1) favors pairs that fill the bin tightly.
    2. Pair-sum normalization: pair_sum/capacity provides a baseline attractiveness
       that scales with how much of the bin is being used.
    
    We blend these: the tight-packing term is the primary driver, but we add
    a small contribution from the normalized pair sum to provide gradient
    information even when remaining space is moderate.
    
    eta[i, j] = 1/(remaining+1) + 0.5 * (pair_sum/capacity) if pair fits
    eta[i, j] = 1e-9 otherwise
    """
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    # Calculate pair sums
    pair_sums = d[:, None] + d[None, :]
    
    # Determine which pairs fit in the bin
    fits = pair_sums <= capacity
    
    # Calculate remaining space for fitting pairs
    remaining = capacity - pair_sums
    
    # Initialize eta
    eta = np.full((n, n), 1e-9, dtype=np.float64)
    
    # For fitting pairs: combine tight-packing inverse with normalized pair sum
    # The inverse remaining term strongly rewards near-perfect fits
    # The pair_sum/capacity term provides a smoother gradient
    tight_reward = 1.0 / (remaining[fits] + 1.0)
    normalized_sum = pair_sums[fits] / capacity
    
    # Blend: primary weight on tight packing, secondary on normalized sum
    eta[fits] = tight_reward + 0.5 * normalized_sum
    
    # Ensure all values are finite
    if not np.all(np.isfinite(eta)):
        eta = np.nan_to_num(eta, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Ensure no negative or zero values (non-positive becomes 1e-9)
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
