"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Normalize each constraint's weights to [0,1], then compute
    heuristic as product of normalized prize and inverse of maximum normalized
    weight across all constraints. This prioritizes items that are highly
    valuable relative to their worst-case resource consumption.
    """
    # prizes shape: (5, 100) -> prize_sum shape: (100,)
    prize_sum = np.sum(prizes, axis=0)
    
    # Normalize each constraint's weights to [0, 1]
    # For each constraint c, normalize weights[c, :] to [0, 1]
    w_min = np.min(weights, axis=1, keepdims=True)  # shape (5, 1)
    w_max = np.max(weights, axis=1, keepdims=True)  # shape (5, 1)
    w_range = w_max - w_min
    # Avoid division by zero
    w_range_safe = np.where(w_range < 1e-12, 1.0, w_range)
    normalized_weights = (weights - w_min) / w_range_safe  # shape (5, 100), values in [0, 1]
    
    # For each item, find the maximum normalized weight across all constraints
    max_norm_weight = np.max(normalized_weights, axis=0)  # shape (100,)
    
    # Compute heuristic: prize_sum / (max_norm_weight + epsilon)
    # This rewards items with high prize that are light in their worst constraint
    heuristic = prize_sum / (max_norm_weight + 1e-6)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
