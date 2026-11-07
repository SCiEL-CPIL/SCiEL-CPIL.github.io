"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed deadline with generous safety margin (evaluator kills at 330s).
_DEADLINE = 330.0
_START_TIME = time.time()


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: numpy array of shape (5, 100) with prizes for each constraint/item.
        weights: numpy array of shape (5, 100) with weights for each constraint/item.
        
    Returns:
        numpy array of shape (100,) with heuristic scores (larger is more attractive).
    """
    # Check deadline (though this function is very fast)
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Fallback to a simple heuristic if we are dangerously close to timeout
        P = np.sum(prizes, axis=0)
        W = np.sum(weights, axis=0)
        return np.maximum(P / (W + 1e-9), 1e-9)

    # 1. Calculate total prize per item
    P = np.sum(prizes, axis=0)
    
    # 2. Calculate cost metrics
    # W_sum: Total resource consumption across all constraints
    W_sum = np.sum(weights, axis=0)
    # W_max: The bottleneck constraint consumption (max weight in any single constraint)
    W_max = np.max(weights, axis=0)
    
    # 3. Combine costs
    # We penalize both total weight and peak weight.
    # W_sum captures the "volume". W_max captures the "shape" inefficiency.
    Cost = W_sum + W_max
    
    # 4. Compute heuristic ratio
    # Value to Cost ratio
    H = P / (Cost + 1e-9)
    
    # 5. Ensure positivity
    H = np.maximum(H, 1e-9)
    
    # 6. Normalize to have mean 1
    # This helps in the first ACO iteration where pheromone is uniform (1.0).
    # If H has a mean of 1, the probability distribution is proportional to H.
    # If H has a mean of 100, it's still proportional, but scaling can affect
    # numerical stability or interaction with pheromone if pheromone changes scale.
    # Here pheromone starts at 1.
    mean_h = np.mean(H)
    if mean_h > 0:
        H = H / mean_h
        
    return H
# EVOLVE-BLOCK-END
