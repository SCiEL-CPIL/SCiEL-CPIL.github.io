"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for each item in a multiple-constraint knapsack.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight costs for each constraint/item
        
    Returns:
        array of shape (100,) - attractiveness score per item (larger = more attractive)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # Maximum weight per item across constraints
    # This identifies the "bottleneck" resource for each item
    max_weight = weights.max(axis=0)
    
    # Heuristic: Ratio of total prize to the maximum weight constraint
    # This favors items with high value that do not heavily strain any single constraint
    epsilon = 1e-9
    heuristic = total_prize / (max_weight + epsilon)
    
    # Ensure positivity and finite values as per task contract
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
