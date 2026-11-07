"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of the
    multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100) with prize values in [0, 1]
        weights: array of shape (5, 100) with weight values in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    # Total prize across all constraints for each item
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Total weight across all constraints for each item
    total_weight = weights.sum(axis=0)  # shape (100,)
    
    # For each constraint, compute the "density" ratio: prize/weight
    # We want items that have high prize relative to their weight in each constraint
    
    # Compute per-constraint efficiency: prize_c / (weight_c + eps)
    eps = 1e-9
    efficiency = prizes / (weights + eps)  # shape (5, 100)
    
    # Combine: we want items that are efficient across all constraints
    # Use geometric mean of efficiencies (more robust to outliers)
    # Or use minimum efficiency (bottleneck constraint matters most)
    
    # Strategy: combine total prize density with minimum per-constraint efficiency
    # This ensures items are good both in aggregate and in the worst constraint
    
    min_efficiency = efficiency.min(axis=0)  # shape (100,)
    total_density = total_prize / (total_weight + eps)  # shape (100,)
    
    # Geometric mean of the two measures, raised to appropriate powers
    # Heuristic = total_density^a * min_efficiency^b
    # Higher a emphasizes overall value, higher b emphasizes constraint balance
    
    # Use log-space for numerical stability
    log_total_density = np.log(total_density + eps)
    log_min_efficiency = np.log(min_efficiency + eps)
    
    # Weighted combination in log space
    # Emphasize overall density slightly more, but strongly consider bottleneck
    combined_log = 0.6 * log_total_density + 0.4 * log_min_efficiency
    
    heuristic = np.exp(combined_log)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
