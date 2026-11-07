"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a heuristic matrix for Ant Colony Optimization.
    
    Strategy:
    1. Feasibility: An edge (i, j) is infeasible if the minimum possible tour length
       d[0, i] + d[i, j] + d[j, 0] exceeds max_len.
    2. Attractiveness: For feasible edges, the heuristic is proportional to the
       "value" of node j divided by the "cost" of visiting j.
       - Value: Prize at j plus a decayed sum of prizes of nearby nodes (potential for future visits).
       - Cost: Distance from i to j plus distance from j to 0 (proxy for return cost).
    """
    start_time = time.time()
    DEADLINE = start_time + 290.0

    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)

    if distance_matrix.shape != (n, n):
        default = np.ones((n, n), dtype=np.float64) * 1e-9
        default[0, 0] = 0.0
        return default

    eps = 1e-9
    sigma = 0.25  # Decay parameter for future potential

    # Precompute distances from/to 0
    d_from_0 = distance_matrix[0, :]
    d_to_0 = distance_matrix[:, 0]

    # Precompute the "local potential" for each node j.
    # P[j] = sum_{k != j} prize[k] * exp(-d[j, k] / sigma)
    # This estimates the value of high-prize nodes near j.
    # We can vectorize this:
    # dist_matrix is (n, n)
    # decayed_weights = exp(-dist_matrix / sigma)
    # Set diagonal to 0 to exclude self
    decayed_weights = np.exp(-distance_matrix / sigma)
    np.fill_diagonal(decayed_weights, 0.0)
    
    # P[j] is the sum over k of prize[k] * decayed_weights[j, k]
    # This is a matrix-vector multiplication: decayed_weights @ prizes
    P = decayed_weights @ prizes
    
    # Numerator for each node j: prize[j] + P[j]
    numerator = prizes + P
    
    # Denominator for edge (i, j): d[i, j] + d[j, 0] + eps
    # d[i, j] is distance_matrix[i, j]
    # d[j, 0] is d_to_0[j]
    # We want a matrix D where D[i, j] = d[i, j] + d_to_0[j]
    denominator = distance_matrix + d_to_0[None, :] + eps
    
    # Calculate raw heuristic
    # H_raw[i, j] = numerator[j] / denominator[i, j]
    H_raw = numerator[None, :] / denominator
    
    # Feasibility mask
    # min_tour_len[i, j] = d[0, i] + d[i, j] + d[j, 0]
    min_tour_len = d_from_0[:, None] + distance_matrix + d_to_0[None, :]
    feasible_mask = min_tour_len <= max_len
    
    # Apply mask
    heuristic = np.where(feasible_mask, H_raw, 1e-9)
    
    # Set diagonal to 1e-9
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
