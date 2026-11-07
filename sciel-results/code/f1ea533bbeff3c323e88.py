"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # Total weight per item (sum across constraints)
    total_weight = weights.sum(axis=0)
    
    # Heuristic: reward high total prize, penalize total weight
    # This aligns with the high-performing baseline heuristic
    h = total_prize / (1.0 + total_weight)
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
