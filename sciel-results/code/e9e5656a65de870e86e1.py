"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Uses normalized weights per constraint to capture relative tightness,
    then combines geometric mean and soft-min of efficiency ratios.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    eps = 1e-9
    
    # Normalize weights per constraint to capture relative tightness
    # Each constraint has capacity 25, so normalize by max weight in that constraint
    max_weights = np.max(weights, axis=1, keepdims=True)  # shape (5, 1)
    normalized_weights = weights / (max_weights + eps)  # shape (5, 100)
    
    # Per-constraint efficiency using normalized weights
    safe_weights = np.maximum(normalized_weights, eps)
    ratios = prizes / safe_weights  # shape (5, 100)
    
    # Geometric mean across constraints - penalizes items heavy in any constraint
    log_efficiency = np.log(np.maximum(ratios, eps))
    geo_mean_efficiency = np.exp(log_efficiency.mean(axis=0))
    
    # Soft-Min approximation to identify the bottleneck constraint efficiency
    # SoftMin(x) = -1/beta * log(sum(exp(-beta * x)))
    # This smooths the minimum, penalizing items with very low efficiency in any single constraint
    beta = 2.0
    
    # For numerical stability, subtract the max of (-beta * ratios) before exp
    v = -beta * ratios
    v_max = np.max(v, axis=0, keepdims=True)
    
    exp_v = np.exp(v - v_max)
    sum_exp = np.sum(exp_v, axis=0)
    
    log_sum_exp = v_max + np.log(sum_exp + eps)
    soft_min_ratios = -1.0 / beta * log_sum_exp
    
    # Combine geometric mean and soft-min
    # Geometric mean captures overall multi-constraint balance
    # Soft-min captures worst-case bottleneck
    # Weight them to balance both concerns
    scores = 0.5 * geo_mean_efficiency + 0.5 * soft_min_ratios
    
    # Ensure all values are positive
    scores = np.maximum(scores, eps)
    
    return scores
# EVOLVE-BLOCK-END
