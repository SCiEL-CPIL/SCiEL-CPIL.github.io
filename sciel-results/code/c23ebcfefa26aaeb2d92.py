"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Uses a 'Prize per Bottleneck Weight' heuristic, which prioritizes items
    that offer high total prize relative to their most constraining weight dimension.
    This addresses the multi-constraint nature by focusing on the resource that
    limits the item's inclusion the most.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # The bottleneck weight for each item is the maximum weight across all constraints.
    # This represents the most significant resource consumption for that item.
    max_weight = weights.max(axis=0)
    
    # Heuristic: Prize divided by the bottleneck weight.
    # This favors items that are "light" in their heaviest dimension relative to their prize.
    # Adding a small epsilon to avoid division by zero.
    h = total_prize / (max_weight + 1e-9)
    
    # Ensure all values are positive and finite
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    return h
# EVOLVE-BLOCK-END
