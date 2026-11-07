"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Combines two efficiency metrics with improved bottleneck handling:
    1. Aggregate Efficiency: Prize per total weight.
    2. Soft-Min Bottleneck: Smooth approximation of the minimum per-constraint ratio.
       Uses log-sum-exp for numerical stability.
    
    The soft-min directly targets the binding constraint, which is more relevant
    for multi-constraint knapsack than geometric mean.
    """
    eps = 1e-9
    
    # 1. Aggregate Efficiency: prizes / sum(weights)
    total_weights = weights.sum(axis=0)
    sum_ratio = prizes / (total_weights + eps)
    
    # 2. Soft-Min Bottleneck: -T * log(mean(exp(-ratios/T)))
    # As T -> 0, this approaches the true minimum ratio.
    # Use T=0.5 for a good balance between smoothness and bottleneck focus.
    ratios = prizes / (weights + eps)
    
    T = 0.5  # Temperature for soft-min
    # Compute -T * log(mean(exp(-ratios/T)))
    # For numerical stability, subtract max before exp
    neg_scaled = -ratios / T
    max_neg_scaled = neg_scaled.max(axis=0, keepdims=True)
    # Clip to avoid overflow in exp
    neg_scaled_clipped = np.clip(neg_scaled - max_neg_scaled, -100, 100)
    exp_terms = np.exp(neg_scaled_clipped)
    mean_exp = exp_terms.mean(axis=0)
    log_mean_exp = np.log(mean_exp + eps)
    soft_min_ratio = -T * (log_mean_exp + max_neg_scaled.squeeze(axis=0))
    
    # Normalize both to mean 1
    sum_ratio_mean = sum_ratio.mean()
    soft_min_mean = soft_min_ratio.mean()
    
    if sum_ratio_mean > 0:
        sum_ratio_norm = sum_ratio / sum_ratio_mean
    else:
        sum_ratio_norm = np.ones_like(sum_ratio)
        
    if soft_min_mean > 0:
        soft_min_norm = soft_min_ratio / soft_min_mean
    else:
        soft_min_norm = np.ones_like(soft_min_ratio)
        
    # Blend: weight more heavily toward bottleneck (structurally more relevant)
    scores = 0.3 * sum_ratio_norm + 0.7 * soft_min_norm
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
