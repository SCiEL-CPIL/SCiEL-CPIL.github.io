"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Uses a refined 'bottleneck slack' approach: for each item, compute the minimum remaining
    capacity ratio across all constraints if that item were added first, then multiply
    by the item's total prize to prioritize items that fit easily into tight constraints.
    Additionally, we normalize by the geometric mean of weights to account for overall
    item "size" and boost items that are lightweight in all constraints.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight values for each constraint/item
    
    Returns:
        array of shape (100,) - attractiveness score per item (larger is more attractive)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = weights.shape
    capacity = 25.0
    
    # Total prize per item (sum across constraints)
    total_prizes = prizes.sum(axis=0)
    
    eps = 1e-9
    
    # Remaining capacity for each item in each constraint
    remaining = capacity - weights  # shape (5, 100)
    
    # Avoid division by zero: use max(w, eps)
    w_safe = np.maximum(weights, eps)
    
    # Ratio of remaining capacity to weight for each constraint/item
    ratios = remaining / w_safe  # shape (5, 100)
    
    # Bottleneck: minimum ratio across constraints for each item
    bottleneck_slack = np.min(ratios, axis=0)  # shape (100,)
    
    # Ensure non-negative (if weight > capacity, remaining is negative, ratio is negative)
    # Clip to a small positive value to maintain positivity
    bottleneck_slack = np.maximum(bottleneck_slack, eps)
    
    # Compute geometric mean of weights per item to penalize heavy items
    # log(weights) can have issues if weights are 0, so use safe weights
    log_w_safe = np.log(w_safe)
    geo_mean_log = np.mean(log_w_safe, axis=0)  # shape (100,)
    geo_mean_weights = np.exp(geo_mean_log)  # shape (100,)
    
    # Scale factor: prefer items with smaller geometric mean weight
    # Use capacity / geo_mean to get a "lightness" bonus
    lightness = capacity / np.maximum(geo_mean_weights, eps)
    
    # Normalize bottleneck_slack to [0, 1] range for better scaling
    slack_min = np.min(bottleneck_slack)
    slack_max = np.max(bottleneck_slack)
    slack_range = slack_max - slack_min
    if slack_range > eps:
        normalized_slack = (bottleneck_slack - slack_min) / slack_range
    else:
        normalized_slack = np.ones_like(bottleneck_slack)
    
    # Normalize lightness similarly
    light_min = np.min(lightness)
    light_max = np.max(lightness)
    light_range = light_max - light_min
    if light_range > eps:
        normalized_light = (lightness - light_min) / light_range
    else:
        normalized_light = np.ones_like(lightness)
    
    # Combine: use weighted geometric mean for smoother combination
    # Heuristic = total_prize * (normalized_slack^0.5) * (normalized_light^0.5)
    # This balances feasibility and lightness while keeping prize as primary driver
    heuristic = total_prizes * np.sqrt(normalized_slack + eps) * np.sqrt(normalized_light + eps)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
