"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of orienteering.
    
    The heuristic is based on the marginal gain of adding node j to the current best
    feasible tour approximation. We estimate the remaining budget capacity as
    max_len - d[i,j] - d[j,0] and weight prize[j] by this slack, thereby dynamically
    prioritizing nodes that maximize prize density per unit of remaining path budget.
    
    Parameters:
    -----------
    prizes : array-like, shape (n,)
        Prize values for each node. Prize at node 0 is 0.
    distance_matrix : array-like, shape (n, n)
        Distance matrix between nodes.
    max_len : float
        Maximum allowed tour length.
    
    Returns:
    --------
    heuristic : ndarray, shape (n, n)
        Edge-attractiveness matrix (larger is more attractive).
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Compute the heuristic matrix
    # For edge (i, j): attractiveness is based on prize[j] weighted by slack
    # slack = max_len - d[i,j] - d[j,0]
    # If slack <= 0, the edge is not feasible in terms of budget, so give it a small value
    
    # d[i,j] is distance_matrix[i, j]
    # d[j,0] is distance_matrix[j, 0]
    
    # Create matrices for broadcasting
    d_ij = distance_matrix  # shape (n, n), d_ij[i, j] = d[i, j]
    d_j0 = distance_matrix[:, 0]  # shape (n,), d_j0[j] = d[j, 0]
    
    # slack[i, j] = max_len - d[i, j] - d[j, 0]
    slack = max_len - d_ij - d_j0[None, :]  # shape (n, n)
    
    # Prize density: prize[j] / (some cost metric)
    # We want to prioritize nodes with high prize and high slack (more budget remaining)
    # A good heuristic: prize[j] * slack[i, j] / (d[i, j] + epsilon)
    # This gives high values for edges that lead to high-prize nodes with lots of slack
    
    eps = 1e-9
    
    # Compute the heuristic
    # For i=0 (start node), we don't have a "current position" cost, so use the same formula
    # prize[j] * max(slack, 0) / (d[i,j] + eps)
    
    # Ensure slack is non-negative for the weighting (if slack is negative, the move is infeasible)
    slack_positive = np.maximum(slack, 0)
    
    # Heuristic: prize[j] * slack_positive[i, j] / (d[i, j] + eps)
    # This emphasizes nodes with high prize and high remaining budget
    
    heuristic = (prizes[None, :] * slack_positive) / (d_ij + eps)
    
    # Also consider a component that rewards direct prize collection
    # Add a small term for prize[j] itself to avoid zeroing out high-prize nodes with low slack
    prize_term = prizes[None, :] / (d_ij + eps)
    
    # Combine: primary is slack-weighted prize, secondary is direct prize density
    heuristic = 0.7 * heuristic + 0.3 * prize_term
    
    # Ensure all entries are positive (non-positive become 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
