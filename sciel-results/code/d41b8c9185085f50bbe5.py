import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Return cost: cost to go from node j back to start
    return_cost = distance_matrix[:, 0]  # shape (n,)
    
    # Local step cost from i to j
    step_cost = distance_matrix  # shape (n, n)
    
    # Total cost of including node j when coming from i: step + return
    total_cost = step_cost + return_cost[None, :]  # shape (n, n)
    
    # Prize per unit total cost (global budget efficiency)
    # Use a small epsilon to avoid division by zero
    eps = 1e-9
    prize_per_cost = (prizes[None, :] + eps) / (total_cost + eps)
    
    # Add a mild local distance preference to break ties
    # and guide the ant to closer nodes among similarly efficient ones
    local_pref = 1.0 / (step_cost + eps)
    
    # Combine: primarily driven by global efficiency, with local distance as tiebreaker
    # Use a small weight for local preference
    heuristic = prize_per_cost * (1.0 + 0.1 * local_pref / (local_pref.max() + eps))
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)
    
    return heuristic
