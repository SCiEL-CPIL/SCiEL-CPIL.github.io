"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    d = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    eps = 1e-9
    
    # Return cost for each node j: d[j, 0]
    return_cost = d[:, 0] + eps
    
    # Heuristic matrix H[i, j]
    # We want to encourage moves to nodes j that have high prize,
    # are close to i (low d[i, j]), and are close to 0 (low d[j, 0]).
    # The term (d[i, j] + d[j, 0]) approximates the additional path length
    # if we go i -> j and then eventually return to 0 (assuming j is the last node,
    # or as a proxy for the "burden" of visiting j).
    
    # Actually, a better proxy for the "cost" of visiting j from i in the context
    # of returning to 0 is d[i, j] + d[j, 0].
    # So, attractiveness = prize_j / (d[i, j] + d[j, 0])
    
    # Vectorized computation:
    # d[i, j] is the matrix itself.
    # d[j, 0] is a vector of length n, which we broadcast along rows.
    # So, denominator = d + return_cost[None, :]
    
    denom = d + return_cost[None, :]
    H = prizes[None, :] / (denom + eps)
    
    # Ensure non-negative and finite
    H = np.maximum(H, 1e-9)
    H = np.where(np.isfinite(H), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
