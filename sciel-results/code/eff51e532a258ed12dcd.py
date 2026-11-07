"""New program for multi-constraint knapsack heuristic scoring."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using multi-criteria ranking.
    
    Approach:
    1. Normalize each constraint's weights independently to [0, 1].
    2. Compute the maximum normalized weight per item.
    3. Compute the total prize per item (sum of prize components) and normalize to [0, 1].
    4. Heuristic = normalized_prize * (1 / (max_norm_weight + epsilon)).
    
    Parameters:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,), one attractiveness score per item
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Step 1: Normalize each constraint's weights to [0, 1]
    w_min = weights.min(axis=1, keepdims=True)
    w_max = weights.max(axis=1, keepdims=True)
    w_range = w_max - w_min
    w_range = np.where(w_range > 1e-15, w_range, 1.0)
    norm_weights = (weights - w_min) / w_range  # shape (5, 100)
    
    # Step 2: Max normalized weight per item
    max_norm_w = norm_weights.max(axis=0)  # shape (100,)
    
    # Step 3: Total prize per item
    # Assuming the objective is to maximize the sum of all prize components
    prize_score = prizes.sum(axis=0)  # shape (100,)
    
    # Normalize prize scores to [0, 1]
    p_min = prize_score.min()
    p_max = prize_score.max()
    p_range = p_max - p_min
    if p_range > 1e-15:
        norm_prize = (prize_score - p_min) / p_range
    else:
        norm_prize = np.ones(100, dtype=np.float64)
    
    # Step 4: Heuristic = norm_prize * (1 / (max_norm_w + eps))
    eps = 1e-6
    inverse_weight = 1.0 / (max_norm_w + eps)
    
    heuristic = norm_prize * inverse_weight
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
