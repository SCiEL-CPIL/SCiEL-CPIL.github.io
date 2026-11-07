"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    eps = 1e-9
    
    # d[j,0]: distance from each node j back to origin
    d_j0 = distance_matrix[:, 0].reshape(1, -1)  # shape (1, n)
    
    # Heuristic: prize / (d[i,j] * d[j,0])
    # This penalizes both the approach distance and the return distance
    # Nodes that are close to current node AND close to origin are most attractive
    numerator = prizes[None, :] + eps
    denominator = (distance_matrix + eps) * (d_j0 + eps)
    
    result = numerator / denominator
    
    # Ensure all values are positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, eps)
    
    return result
# EVOLVE-BLOCK-END
