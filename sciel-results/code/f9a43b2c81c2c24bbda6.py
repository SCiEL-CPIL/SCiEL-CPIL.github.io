"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix):
    """
    Computes an edge-attractiveness matrix for ACO.
    Uses a combination of inverse distance and exponential decay to
    strongly favor short edges while maintaining numerical stability.
    """
    eps = 1e-9
    d = distance_matrix + eps
    
    # Use exponential decay: exp(-alpha * d) gives a smooth, bounded
    # attractiveness that strongly favors short edges
    # alpha controls the "steepness" - larger means stronger preference for short edges
    alpha = 5.0
    exp_component = np.exp(-alpha * d)
    
    # Combine with inverse distance for additional emphasis on very short edges
    inv_component = 1.0 / d
    
    # Weighted combination
    heuristic = 0.5 * exp_component + 0.5 * inv_component / np.max(inv_component)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
