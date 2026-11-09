import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    # Return cost to origin for each node: d[j, 0]
    return_cost = distance_matrix[:, 0].copy()
    
    # Total cost of going from i to j and then back to 0: d[i,j] + d[j,0]
    # Shape: (n, n) where entry [i,j] = d[i,j] + d[j,0]
    total_cost = distance_matrix + return_cost[None, :]
    
    # Heuristic: prize density = prizes[j] / total_cost[i,j]
    # This is asymmetric: entry [i,j] depends on where you are (i) and where you go (j)
    # Ensure no division by zero
    eps = 1e-9
    total_cost_safe = np.maximum(total_cost, eps)
    
    # prizes[j] broadcast over rows: shape (1, n)
    prize_col = prizes[None, :]
    
    heuristic = prize_col / total_cost_safe
    
    # Ensure all values are positive; replace non-positive with 1e-9
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
