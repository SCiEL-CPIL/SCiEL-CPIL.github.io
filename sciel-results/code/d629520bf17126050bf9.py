"""Rewritten program for the suite task; new heuristic design."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Approach: Multi-criteria ranking via normalized weights.
    - Normalize each constraint's weight independently to [0, 1].
    - For each item, find the maximum normalized weight across all constraints (worst-case resource consumption).
    - Heuristic = product of (normalized prize) and (inverse of max normalized weight + epsilon).
    - This prioritizes items that are highly valuable relative to their worst-case resource usage.
    
    Parameters:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,), one attractiveness score per item (larger is more attractive)
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Normalize each constraint's weights independently to [0, 1]
    # For each row (constraint), scale weights to [0, 1]
    w_min = weights.min(axis=1, keepdims=True)
    w_max = weights.max(axis=1, keepdims=True)
    w_range = w_max - w_min
    # Avoid division by zero
    w_range = np.where(w_range > 0, w_range, 1.0)
    normalized_weights = (weights - w_min) / w_range
    
    # For each item, find the maximum normalized weight across all constraints
    # This represents the worst-case resource consumption
    max_norm_weight = normalized_weights.max(axis=0)  # shape (100,)
    
    # Normalize prizes: use the mean prize per item across constraints as a proxy for total value
    # Or we could use the sum or max. Let's use the sum for total prize value.
    item_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Normalize item prizes to [0, 1]
    p_min = item_prize.min()
    p_max = item_prize.max()
    p_range = p_max - p_min
    if p_range > 0:
        normalized_prize = (item_prize - p_min) / p_range
    else:
        normalized_prize = np.ones_like(item_prize)
    
    # Heuristic: product of normalized prize and inverse of (max normalized weight + epsilon)
    # Items with high prize and low worst-case weight get high scores
    epsilon = 1e-9
    heuristic = normalized_prize / (max_norm_weight + epsilon)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
