"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Static opportunity-cost heuristic for ant colony orienteering.
    
    H[i, j] = prize[j] * (1 - (d[i,j] + d[j,0]) / max_len)
    
    This rewards high-prize destinations that consume little of the length budget,
    and penalizes (or zeros out) moves that would leave insufficient slack for
    the return trip to the depot.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    eps = 1e-9
    
    # d[j, 0] for each j
    return_cost = distance_matrix[:, 0]
    
    # Total cost of going i -> j -> 0
    total_cost = distance_matrix + return_cost[None, :]
    
    # Fraction of max_len consumed
    fraction = total_cost / max_len
    
    # Heuristic: prize[j] * remaining budget fraction
    heuristic = prizes[None, :] * (1.0 - fraction)
    
    # Clamp to small positive for infeasible or zero-valued entries
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
