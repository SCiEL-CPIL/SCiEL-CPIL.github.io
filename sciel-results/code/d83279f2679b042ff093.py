"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Combines two robust efficiency metrics:
    1. Aggregate Efficiency: Prize per total weight.
       Captures overall resource usage.
    2. Bottleneck Efficiency: Geometric mean of per-constraint prize/weight ratios.
       Captures the constraint that limits the item the most (smooth min).
    
    Both metrics are normalized to have a mean of 1 across items to ensure
    they contribute equally to the final heuristic score, preventing scale
    dominance by either metric.
    """
    eps = 1e-9
    
    # Per-constraint prize/weight ratio for each item
    # Shape: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Aggregate efficiency: total prize / total weight
    total_prizes = prizes.sum(axis=0)
    total_weights = weights.sum(axis=0)
    agg_eff = total_prizes / (total_weights + eps)
    
    # Bottleneck efficiency: geometric mean of per-constraint ratios
    # This is a smooth surrogate for the minimum ratio (bottleneck constraint)
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_mean = np.exp(log_geo_mean)
    
    # Normalize both metrics to have mean 1 across items
    # This ensures they contribute equally regardless of absolute scale
    agg_eff_norm = agg_eff / (np.mean(agg_eff) + eps)
    geo_mean_norm = geo_mean / (np.mean(geo_mean) + eps)
    
    # Combine with equal weights
    # The geometric mean captures cross-constraint efficiency (bottleneck-aware)
    # The aggregate efficiency captures overall resource usage
    combined = 0.5 * agg_eff_norm + 0.5 * geo_mean_norm
    
    # Ensure all values are positive
    scores = np.maximum(combined, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
