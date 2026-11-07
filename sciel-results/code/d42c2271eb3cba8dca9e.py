"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Combines three robust efficiency metrics normalized to mean 1:
    1. Aggregate Efficiency: Total Prize / Total Weight.
    2. Bottleneck Efficiency: Total Prize / Max Weight (per constraint).
    3. Geometric Mean Efficiency: Geometric mean of (Prize_c / Weight_c) for each constraint.
    
    This balances overall density, the most restrictive constraint, and balanced efficiency.
    """
    start_time = time.time()
    
    # Ensure inputs are float64 arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    eps = 1e-9
    
    # 1. Calculate Total Prize per item
    # Assuming the objective is to maximize the sum of all prize components
    item_prizes = prizes.sum(axis=0)
    
    # 2. Aggregate Efficiency: Prize / Sum(Weights)
    total_weights = weights.sum(axis=0)
    agg_ratio = item_prizes / (total_weights + eps)
    
    # 3. Bottleneck Efficiency: Prize / Max(Weights)
    # The item is limited by the constraint where it has the highest weight
    max_weights = weights.max(axis=0)
    bott_ratio = item_prizes / (max_weights + eps)
    
    # 4. Geometric Mean Efficiency
    # Per-constraint ratio: Prize_c / Weight_c
    # We use the raw prizes here to see if an item is efficient in each specific constraint
    # If prizes are identical across constraints, this is equivalent to Prize / GeometricMean(Weights)
    ratios = prizes / (weights + eps)
    
    # Geometric mean in log-space for stability
    # Clip ratios to avoid log(0) or negative values (though inputs are positive)
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_ratio = np.exp(log_geo_mean)
    
    # Normalize each metric to have a mean of 1
    # This ensures they contribute equally regardless of scale
    def normalize(arr):
        mean_val = np.mean(arr)
        if mean_val > 0:
            return arr / mean_val
        else:
            return np.ones_like(arr)
            
    agg_norm = normalize(agg_ratio)
    bott_norm = normalize(bott_ratio)
    geo_norm = normalize(geo_ratio)
    
    # Combine the metrics
    # Weighting: 
    # Bottleneck is often the most critical in multi-constraint knapsack.
    # Aggregate gives a good baseline density.
    # Geo mean provides a smooth balance.
    # Weights sum to 1.0
    weights_mix = np.array([0.4, 0.3, 0.3]) # [agg, bott, geo]
    
    scores = (weights_mix[0] * agg_norm + 
              weights_mix[1] * bott_norm + 
              weights_mix[2] * geo_norm)
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    # Time guard
    if time.time() - start_time > 290:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
