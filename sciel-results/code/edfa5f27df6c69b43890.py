"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Key insight: decouple node value (prize minus return penalty) from step cost.
    # The return cost d[j,0] is a property of node j (how expensive it is to include
    # in a path that must return to 0), while d[i,j] is the local step cost.
    # By separating these, we avoid the saturation that occurs when both are in
    # the same denominator power.
    
    d_j0 = distance_matrix[:, 0]  # (n,) return cost for each node
    
    # Net node value: prize minus a penalty for return distance
    # This makes high-prize nodes far from origin less attractive
    # Use a moderate penalty so that geometry matters but doesn't dominate
    return_penalty = 0.5
    net_value = prizes - return_penalty * d_j0
    
    # Clip to ensure non-negative (a node with prize < penalty*d[j,0] gets 0 value)
    # But keep a small floor so the heuristic doesn't become exactly 0
    eps = 1e-9
    net_value = np.maximum(net_value, eps)
    
    # Step cost: d[i, j] - the local traversal cost
    # Use a small epsilon to avoid division by zero on diagonal
    step_cost = distance_matrix + eps
    
    # Heuristic: net_value_j^alpha / step_cost[i,j]^beta
    # alpha > 1 to emphasize high net-value nodes
    # beta ~ 1 to penalize long steps but not too aggressively
    alpha = 2.0
    beta = 1.0
    
    prize_term = np.power(net_value[None, :], alpha)  # (1, n)
    cost_term = np.power(step_cost, beta)  # (n, n)
    
    heuristic = prize_term / cost_term
    
    # Row-normalize to have mean 1 per row for consistent scale
    # This helps the ACO probability selection be well-calibrated
    row_means = np.mean(heuristic, axis=1, keepdims=True)
    row_means = np.maximum(row_means, eps)
    heuristic = heuristic / row_means
    
    # Set diagonal to very small (no self-moves)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
