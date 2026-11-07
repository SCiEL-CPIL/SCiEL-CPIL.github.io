"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Build the heuristic matrix using inverse squared waste
    pair_sum = d[:, None] + d[None, :]
    fits = pair_sum <= capacity
    
    # For fitting pairs, compute waste and use inverse squared waste as attractiveness
    waste = capacity - pair_sum
    # Only consider fitting pairs; set waste to a large value for non-fitting to avoid division issues
    safe_waste = np.where(fits, waste, capacity)
    
    # Inverse squared waste: 1 / (waste^2 + epsilon) to avoid division by zero
    epsilon = 1e-6
    base_score = np.where(fits, 1.0 / (safe_waste ** 2 + epsilon), 0.0)
    
    # Scale up to make the heuristic more pronounced
    base_score = base_score * 100.0
    
    # Ensure non-finite values are replaced
    result = np.where(np.isfinite(base_score), base_score, 1e-9)
    
    # Ensure all non-positive become 1e-9
    result = np.where(result > 0, result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
