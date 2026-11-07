"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Uses a refined heuristic that combines:
    1. Geometric mean of per-constraint prize/weight ratios (efficiency across all constraints)
    2. A penalty for items that are close to filling any single constraint
    3. A bonus for items with high absolute prize value
    """
    eps = 1e-9
    capacity = 25.0
    
    # Per-constraint prize/weight ratio for each item
    # Shape: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Geometric mean across constraints: (prod(ratios_c))^(1/5)
    # Using log-space for numerical stability
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_mean = np.exp(log_geo_mean)
    
    # Compute the maximum weight fraction across all constraints for each item
    # This indicates how "heavy" an item is relative to capacity in its worst constraint
    max_weight_fraction = weights.max(axis=0) / capacity  # Shape: (100,)
    
    # Penalty for items that consume a large fraction of any constraint
    # Items that use more than 50% of a constraint's capacity get penalized
    # This helps avoid items that would block other good items
    penalty = 1.0 / (1.0 + 2.0 * max_weight_fraction)
    
    # Absolute prize score: items with higher total prize are more attractive
    total_prize = prizes.sum(axis=0)  # Shape: (100,)
    prize_score = total_prize / (total_prize.mean() + eps)
    
    # Combine all components
    # Weighted combination: 0.5 * geo_mean + 0.3 * geo_mean * penalty + 0.2 * prize_score
    combined = 0.5 * geo_mean + 0.3 * geo_mean * penalty + 0.2 * prize_score
    
    # Normalize to have mean ~1 for stable blending with pheromone
    combined = combined / (combined.mean() + eps)
    
    # Ensure all values are positive
    combined = np.maximum(combined, 1e-9)
    
    return combined
# EVOLVE-BLOCK-END
