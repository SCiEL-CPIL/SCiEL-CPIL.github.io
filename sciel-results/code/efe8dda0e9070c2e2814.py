import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization.
    
    The heuristic is designed to balance the prize value of a node against
    the cost of visiting it, considering the total detour distance.
    
    Formula:
    eta[i, j] = (prize[j] + eps)^alpha / ((d[i, j] + d[j, 0] + eps)^beta)
    
    This favors nodes with high prizes that are close to the current node 
    and close to the origin (minimizing the total path extension).
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Parameters
    # Higher alpha emphasizes prize value
    # Higher beta emphasizes distance penalty
    alpha = 4.0
    beta = 2.0
    eps = 1e-9
    
    # Extract distance to origin for each node
    # distance_matrix[j, 0] is the distance from j to 0
    dist_to_origin = distance_matrix[:, 0]
    
    # Calculate heuristic
    # prize_term: shape (N,) -> broadcast to (N, N) along axis 0
    prize_term = (prizes + eps) ** alpha
    
    # detour_term: shape (N, N)
    # d[i, j] + d[j, 0]
    # dist_to_origin is (N,), we need it to be (1, N) to broadcast across rows
    # distance_matrix is (N, N)
    # dist_to_origin[None, :] is (1, N)
    detour = distance_matrix + dist_to_origin[None, :]
    
    detour_term = (detour + eps) ** beta
    
    heuristic = prize_term[None, :] / detour_term
    
    # Ensure validity: positive and finite
    # If a node has 0 prize, heuristic is 0. 
    # The problem says "Heuristic entries that are not positive become 1e-9".
    # So we replace <= 0 with 1e-9.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
