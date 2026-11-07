"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of orienteering.
    
    The heuristic guides ants toward high-prize nodes that are both nearby and
    close to the origin, using a dual-distance penalty structure.
    
    Strategy:
    - For each edge (i, j), the heuristic is proportional to prize[j] divided by
      a cost term that accounts for both the travel distance d[i,j] and the return
      distance d[j,0].
    - A feasibility mask ensures that only edges leading to nodes that can still
      be visited (given remaining length budget) get high heuristic values.
    - We use a smooth approximation to avoid discontinuities.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    n = len(prizes)
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure prize at node 0 is 0
    prizes = prizes.copy()
    prizes[0] = 0.0

    # Compute the dual-distance cost: d[i,j] * d[j,0]
    # This captures the true cost of visiting node j: you must travel to it AND return
    d_j_0 = distance_matrix[:, 0]  # distance from each node to origin
    # cost[i,j] = d[i,j] * d[j,0] + d[i,j] + d[j,0] (combined penalty)
    # Actually, let's think more carefully. The constraint is:
    # length_so_far + d[i,j] + d[j,0] <= max_len
    # So the "cost" of visiting j from i is d[i,j] + d[j,0]
    # But we want to penalize edges where this sum is large relative to max_len
    
    # Feasibility: from node i, can we visit node j and still return?
    # The remaining budget from node i (assuming we've used some length) is tricky
    # because it depends on path length. But as a static heuristic, we can use:
    # An edge (i,j) is "feasible" if d[i,j] + d[j,0] <= max_len (optimistic: assume
    # we're at the start of the path from i)
    
    # More refined: the worst case is that we've already used max_len - d[i,0] - d[i,j] - d[j,0]
    # But for a static heuristic, let's use the constraint d[i,j] + d[j,0] <= max_len
    # as a soft feasibility indicator.
    
    # Compute cost matrix: cost[i,j] = d[i,j] + d[j,0]
    cost = distance_matrix + d_j_0[None, :]
    
    # Feasibility mask: edges where cost <= max_len are more attractive
    # Use a smooth sigmoid-like transition
    # Scale: edges with cost close to max_len are less attractive
    # Use: feasibility_factor = 1 / (1 + exp(alpha * (cost - max_len)))
    # This gives ~1 when cost << max_len, ~0.5 when cost = max_len, ~0 when cost >> max_len
    
    alpha = 5.0  # sharpness of the feasibility transition
    # Avoid overflow in exp
    x = alpha * (cost - max_len)
    x = np.clip(x, -50, 50)
    feasibility_factor = 1.0 / (1.0 + np.exp(x))
    
    # Base heuristic: prize[j] / (cost + epsilon)
    epsilon = 1e-9
    base_heuristic = prizes[None, :] / (cost + epsilon)
    
    # Apply feasibility factor
    heuristic = base_heuristic * feasibility_factor
    
    # Add a small regularization to avoid zeros
    heuristic = heuristic + 1e-9
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check for non-finite values
    if not np.all(np.isfinite(heuristic)):
        # Replace non-finite values
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
