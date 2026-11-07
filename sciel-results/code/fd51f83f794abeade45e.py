"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: Array of shape (C, N) or (N,), where C is constraints and N is items.
        weights: Array of shape (C, N).
        
    Returns:
        Array of shape (N,) with heuristic scores.
    """
    # Ensure prizes is treated as having the same number of rows as weights if needed for broadcasting
    if prizes.ndim == 1:
        values = prizes
    else:
        values = prizes.sum(axis=0)
        
    # Add epsilon to weights to avoid division by zero
    eps = 1e-9
    safe_weights = weights + eps
    
    # Calculate ratio of value to weight for each constraint
    # ratios has shape (C, N)
    ratios = values / safe_weights
    
    # Soft-min via log-sum-exp for a smooth bottleneck penalty
    # Temperature parameter controls smoothness: lower = closer to hard min
    temperature = 0.1
    
    # Soft-min: -T * log(sum(exp(-x/T)))
    # For numerical stability, subtract max before exp
    neg_ratios = -ratios
    max_neg = np.max(neg_ratios, axis=0, keepdims=True)
    exp_neg = np.exp((neg_ratios - max_neg) / temperature)
    sum_exp = np.sum(exp_neg, axis=0)
    
    # Soft-min is -T * (log(sum_exp) + max_neg/T) = -T * log(sum_exp) - max_neg
    soft_min_scores = -temperature * np.log(sum_exp) - max_neg.squeeze(axis=0)
    
    # Geometric alignment with the efficient frontier direction
    # Compute mean prize per unit weight across all constraints (the "efficient frontier" direction)
    mean_ratio_per_constraint = np.mean(ratios, axis=1)  # shape (C,)
    
    # For each item, compute cosine similarity between its ratio vector and the mean ratio vector
    # ratios has shape (C, N), mean_ratio_per_constraint has shape (C,)
    item_norms = np.sqrt(np.sum(ratios ** 2, axis=0))  # shape (N,)
    mean_norm = np.sqrt(np.sum(mean_ratio_per_constraint ** 2))  # scalar
    
    # Cosine similarity: dot(ratios[:, i], mean) / (|ratios[:, i]| * |mean|)
    dot_products = np.sum(ratios * mean_ratio_per_constraint[:, np.newaxis], axis=0)  # shape (N,)
    cosine_sim = dot_products / (item_norms * mean_norm + eps)  # shape (N,)
    
    # Clip cosine similarity to [0, 1] to ensure positivity
    cosine_sim = np.clip(cosine_sim, 0, 1)
    
    # Combine soft-min score with geometric alignment
    # Use a geometric blend: geometric mean of soft_min and a scaled cosine alignment
    # Scale cosine to be comparable in magnitude to soft_min_scores
    # Normalize soft_min_scores to [0, 1] range for blending
    sm_min = np.min(soft_min_scores)
    sm_max = np.max(soft_min_scores)
    sm_range = sm_max - sm_min + eps
    soft_min_norm = (soft_min_scores - sm_min) / sm_range
    
    # Blend: higher cosine similarity boosts items that are proportionally efficient
    # Use a weighted geometric mean
    alpha = 0.3  # weight for geometric alignment
    blended = (soft_min_norm ** (1 - alpha)) * (cosine_sim ** alpha)
    
    # Scale back to original magnitude range
    final_scores = sm_min + blended * sm_range
    
    # Ensure scores are positive for the ACO
    # Replace any non-finite or non-positive values with a small number
    scores = np.where(np.isfinite(final_scores) & (final_scores > 0), final_scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
