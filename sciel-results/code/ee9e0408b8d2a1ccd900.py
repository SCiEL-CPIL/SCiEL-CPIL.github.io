"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Structural approach: Use a multi-scale bottleneck detection that combines
    geometric mean (overall balance) with a tuned soft-min (worst-case penalty),
    where the soft-min uses a higher sensitivity to better identify items that
    would quickly saturate any single constraint.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    eps = 1e-9
    
    # Per-constraint efficiency: prize / weight for each constraint
    safe_weights = np.maximum(weights, eps)
    ratios = prizes / safe_weights  # shape (5, 100)
    
    # Geometric mean across constraints - captures overall multi-constraint balance
    log_efficiency = np.log(np.maximum(ratios, eps))
    geo_mean_efficiency = np.exp(log_efficiency.mean(axis=0))
    
    # Multi-scale soft-min: use two different beta values to capture
    # both moderate and severe bottlenecks
    # Higher beta = more sensitive to the minimum (penalizes worst constraint more)
    
    # Moderate sensitivity (beta=3.0)
    beta1 = 3.0
    v1 = -beta1 * ratios
    v1_max = np.max(v1, axis=0, keepdims=True)
    exp_v1 = np.exp(v1 - v1_max)
    sum_exp1 = np.sum(exp_v1, axis=0)
    log_sum_exp1 = v1_max + np.log(sum_exp1 + eps)
    soft_min_1 = -1.0 / beta1 * log_sum_exp1
    
    # High sensitivity (beta=6.0) - more focused on the single worst constraint
    beta2 = 6.0
    v2 = -beta2 * ratios
    v2_max = np.max(v2, axis=0, keepdims=True)
    exp_v2 = np.exp(v2 - v2_max)
    sum_exp2 = np.sum(exp_v2, axis=0)
    log_sum_exp2 = v2_max + np.log(sum_exp2 + eps)
    soft_min_2 = -1.0 / beta2 * log_sum_exp2
    
    # Combine: geometric mean for overall balance, weighted soft-mins for bottleneck detection
    # The high-sensitivity soft-min gets more weight to strongly penalize items
    # that are very inefficient in at least one constraint
    scores = 0.3 * geo_mean_efficiency + 0.3 * soft_min_1 + 0.4 * soft_min_2
    
    # Ensure all values are positive
    scores = np.maximum(scores, eps)
    
    return scores
# EVOLVE-BLOCK-END
