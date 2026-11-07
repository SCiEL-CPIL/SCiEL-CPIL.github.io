"""Heuristic matrix for ant colony optimization of orienteering - static analytical approach."""
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a static heuristic matrix for the ACO algorithm.
    
    The heuristic is defined as:
    eta[i, j] = prizes[j] / (distance_matrix[i, j] + distance_matrix[j, 0])^alpha
    
    This rewards nodes with high prizes and penalizes moves that are expensive
    in terms of the distance to the node plus the distance back to the start,
    which is the critical cost component for the orienteering constraint.
    
    Parameters:
    - prizes: array of prizes for each node.
    - distance_matrix: NxN matrix of distances.
    - max_len: maximum path length (not directly used in the static formula, 
               but defines the problem context).
    
    Returns:
    - eta: NxN heuristic matrix.
    """
    N = len(prizes)
    d_j0 = distance_matrix[:, 0]
    
    # Exponent for distance penalty
    # 1.0 is linear (prize/cost), 1.5 or 2.0 penalizes distance more heavily.
    # 1.5 is a robust choice for balancing exploration of high-value vs low-cost nodes.
    alpha = 1.5
    eps = 1e-9
    
    # Calculate effective cost for moving from i to j:
    # It's the distance i->j plus the distance j->0 (since we must eventually return to 0).
    # This "round-trip" cost is a good proxy for the budget consumption.
    cost = distance_matrix + d_j0[None, :]
    
    # Ensure no division by zero
    cost = np.maximum(cost, eps)
    
    # Calculate heuristic: prize of destination j divided by cost^alpha
    # prizes[None, :] broadcasts to (N, N)
    eta = (prizes[None, :] / (cost ** alpha))
    
    # Ensure all entries are positive and finite
    # If a prize is 0, eta becomes 0. The prompt says non-positive become 1e-9.
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
