"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a greedy ratio heuristic matrix for ant colony optimization.
    
    The heuristic is:
        eta[i, j] = (pi_j + eps) / (d_ij + d_j0 + eps)
    
    This maximizes the immediate prize density relative to the length cost of the move and return trip.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    
    # Ensure prizes[0] is 0 (depot has no prize)
    prizes = prizes.copy()
    prizes[0] = 0.0
    
    # Compute the "cost" of visiting node j from any node i and returning to depot:
    # cost[i, j] = d[i, j] + d[j, 0]
    d_to_depot = distance_matrix[:, 0]  # shape (n,)
    
    # cost[i, j] = d[i, j] + d[j, 0]
    cost = distance_matrix + d_to_depot[None, :]  # shape (n, n)
    
    eps = 1e-9
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, eps)
    
    # Prize vector (broadcast over rows)
    prize_vec = prizes[None, :] + eps  # shape (1, n)
    
    # Compute the heuristic: Greedy Ratio
    eta = prize_vec / cost_safe
    
    # For the diagonal (i == j), the ant can't move to itself.
    # Set to a very small value to discourage self-loops.
    np.fill_diagonal(eta, eps)
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    # Clamp to avoid extreme values
    eta = np.clip(eta, eps, 1e6)
    
    return eta
# EVOLVE-BLOCK-END
