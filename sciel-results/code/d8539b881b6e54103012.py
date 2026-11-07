"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using prize-per-total-weight ratio with
    a flexibility bonus for items whose weight vectors are balanced across
    constraints (low variance), since such items fit more partial solutions.
    """
    # Base heuristic: total prize divided by total weight across constraints
    total_prize = prizes.sum(axis=0)
    total_weight = weights.sum(axis=0)
    
    # Base score: prize density
    base_score = total_prize / (total_weight + 1e-9)
    
    # Flexibility bonus: items with more balanced weight distribution
    # across constraints are more versatile in a multi-constraint setting.
    # Compute variance of weight vector per item (lower variance = more balanced)
    weight_mean = weights.mean(axis=0, keepdims=True)
    weight_var = np.var(weights, axis=0)  # shape (100,)
    
    # Normalize variance to [0, 1] range for scaling
    var_min = weight_var.min()
    var_max = weight_var.max()
    if var_max > var_min:
        normalized_var = (weight_var - var_min) / (var_max - var_min)
    else:
        normalized_var = np.zeros_like(weight_var)
    
    # Flexibility score: 1 - normalized_var (higher for balanced items)
    flexibility = 1.0 - normalized_var
    
    # Combine: base score with a small flexibility modulation
    # The flexibility factor ranges from 0.9 to 1.1 (mild adjustment)
    flex_factor = 0.9 + 0.2 * flexibility
    
    h = base_score * flex_factor
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
