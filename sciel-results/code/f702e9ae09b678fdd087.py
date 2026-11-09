"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    pair = d[:, None] + d[None, :]
    mask = pair <= capacity
    
    # Complementarity graph approach:
    # Reward pairs that leave residual capacity that is a near-integer multiple
    # of the smallest item size, promoting structural fit rather than just best-fit.
    min_item = d.min()
    
    # Residual capacity after placing items i and j together
    residual = capacity - pair  # shape (n, n)
    
    # For feasible pairs, compute how well the residual aligns with multiples of min_item
    # We want residual to be close to k * min_item for some integer k >= 0
    # This means residual / min_item should be close to an integer
    
    # Only consider feasible pairs
    feasible_residual = np.where(mask, residual, 0.0)
    
    # Normalize by min_item
    normalized_residual = feasible_residual / min_item
    
    # Distance to nearest integer (0 means perfect alignment)
    nearest_int = np.round(normalized_residual)
    distance_to_int = np.abs(normalized_residual - nearest_int)
    
    # Score: lower distance = better structural fit
    # Use exponential decay to strongly favor near-perfect alignment
    structural_score = np.exp(-5.0 * distance_to_int)
    
    # Also reward larger fill ratios (tighter packing)
    ratio = np.where(mask, pair / capacity, 0.0)
    fill_score = ratio ** 8
    
    # Combine structural and fill scores
    # Structural fit is primary, fill ratio is secondary
    heuristic = structural_score * (0.5 + 0.5 * fill_score)
    
    # Boost by the size of item j to prefer placing larger items (more efficient)
    size_boost = 1.0 + 0.3 * (d[None, :] - min_item) / (d.max() - min_item + 1e-10)
    heuristic = heuristic * size_boost
    
    # Ensure non-positive/infeasible entries become 1e-9
    heuristic = np.where(mask & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
