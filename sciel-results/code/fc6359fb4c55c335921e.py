import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness matrix for ant colony orienteering.
    
    heuristic[i, j] = (prize[j] + eps) / (d[i,j] + d[j,0] + eps)^alpha
    
    This favors edges to high-prize nodes with low marginal cost
    (distance from i to j plus return distance from j to origin).
    """
    eps = 1e-9
    alpha = 1.5
    
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Marginal cost of visiting j from i: d[i,j] + d[j,0]
    # d[j,0] is the distance from j back to origin
    d_to_origin = distance_matrix[:, 0]  # shape (n,)
    marginal_cost = distance_matrix + d_to_origin[None, :]  # shape (n, n)
    
    # Heuristic: prize at destination / marginal_cost^alpha
    prize_term = prizes[None, :] + eps  # shape (1, n)
    cost_term = (marginal_cost + eps) ** alpha  # shape (n, n)
    
    heur = prize_term / cost_term
    
    # Ensure positive and finite
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    return heur
