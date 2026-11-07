"""Heuristic matrix for ant colony optimization of orienteering."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a regret-based heuristic matrix for ACO orienteering.
    
    The heuristic eta[i, j] is designed to bias ants toward moves that
    maximize immediate prize gain while minimizing opportunity cost.
    
    Strategy:
    - For each pair (i, j), compute the 'regret' of moving to j:
      regret[i, j] = max_{k != i, j, k reachable} prize[k] - prize[j]
      (clamped to be non-negative)
    - The attractiveness is proportional to prize[j] / (distance[i,j] + distance[j,0])
      but penalized by the regret term.
    - Specifically: eta[i, j] = (prize[j] + eps) / ((distance[i,j] + distance[j,0] + eps) * (1 + regret[i,j] * alpha))
      where alpha controls the strength of the regret penalty.
    
    This makes moves to high-prize, low-distance nodes more attractive,
    especially when there are other high-prize nodes that could be reached instead.
    """
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure non-negative distances
    distance_matrix = np.maximum(distance_matrix, 0.0)
    
    eps = 1e-9
    alpha = 0.5  # Regret penalty strength
    
    # Precompute: for each node j, the return distance d[j, 0]
    d_to_origin = distance_matrix[:, 0]  # shape (n,)
    
    # For each node i, determine which nodes j are reachable from i
    # given the length constraint: d[i,j] + d[j,0] <= max_len
    # (We ignore the current path length for the heuristic, as it's static)
    
    # Compute base efficiency: prize[j] / (d[i,j] + d[j,0])
    # Shape: (n, n)
    detour_cost = distance_matrix + d_to_origin[None, :]  # d[i,j] + d[j,0]
    base_efficiency = (prizes[None, :] + eps) / (detour_cost + eps)
    
    # For each i, find the maximum prize among all reachable nodes k (k != i)
    # A node k is reachable from i if d[i,k] + d[k,0] <= max_len
    reachable_mask = (detour_cost <= max_len)  # shape (n, n)
    # Exclude self
    for i in range(n):
        reachable_mask[i, i] = False
    
    # For each i, find the max prize among reachable nodes
    # We'll compute regret[i, j] = max_{k reachable from i, k != j} prize[k] - prize[j]
    # Clamped to >= 0
    
    regret = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        # Get all reachable nodes from i
        reachable_nodes = np.where(reachable_mask[i, :])[0]
        if len(reachable_nodes) == 0:
            continue
        
        # Get prizes of reachable nodes
        reachable_prizes = prizes[reachable_nodes]
        
        # For each reachable node j, compute regret
        for idx, j in enumerate(reachable_nodes):
            # Max prize among other reachable nodes (excluding j)
            if len(reachable_nodes) > 1:
                # Create a copy of reachable_prizes with j's prize set to -inf
                other_prizes = reachable_prizes.copy()
                other_prizes[idx] = -np.inf
                max_other_prize = np.max(other_prizes)
            else:
                max_other_prize = 0.0
            
            regret[i, j] = max(0.0, max_other_prize - prizes[j])
    
    # Compute final heuristic
    # eta[i, j] = base_efficiency[i, j] / (1 + alpha * regret[i, j])
    # But only for reachable nodes; for unreachable nodes, set to a very small value
    
    heuristic = base_efficiency / (1.0 + alpha * regret)
    
    # Set unreachable nodes to a very small value
    heuristic[~reachable_mask] = eps
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)
    
    return heuristic
# EVOLVE-BLOCK-END
