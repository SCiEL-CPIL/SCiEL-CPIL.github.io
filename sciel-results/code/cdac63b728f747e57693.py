"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering.
    
    Design:
    - Uses a 'Roundtrip Cost' heuristic.
    - The cost of moving i->j is approximated by d[i,j] + d[j,0] (cost to go to j and return to start).
    - Attractiveness is Prize / (Roundtrip Cost)^2.
    - This strongly penalizes moves to nodes that are far from the start, even if they are close to the current node,
      preventing ants from getting trapped in distant clusters with insufficient length to return or visit others.
    - Normalized to max value 1 for stability.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # epsilon to avoid division by zero and ensure positivity
    eps = 1e-6
    
    # Calculate roundtrip cost matrix: M[i, j] = d[i, j] + d[j, 0]
    # d[j, 0] is the distance from node j back to the start (node 0)
    d_to_start = distance_matrix[:, 0]
    # d_to_start has shape (n,). 
    # We want M[i, j] = distance_matrix[i, j] + d_to_start[j]
    # distance_matrix is (n, n). d_to_start[None, :] is (1, n).
    roundtrip_cost = distance_matrix + d_to_start[None, :]
    
    # Heuristic: Prize / (Roundtrip Cost)^2
    # prizes[None, :] is (1, n), broadcasts to (n, n)
    numerator = prizes[None, :] + eps
    denominator = (roundtrip_cost + eps) ** 2
    
    eta = numerator / denominator
    
    # Ensure all values are finite and positive
    eta[~np.isfinite(eta)] = eps
    eta[eta <= 0] = eps
    
    # Normalize to [0, 1] range (max becomes 1)
    max_eta = np.max(eta)
    if max_eta > 0:
        eta = eta / max_eta
    else:
        eta = np.ones_like(eta) * eps
        
    return eta
# EVOLVE-BLOCK-END
