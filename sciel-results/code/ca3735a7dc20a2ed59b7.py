"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search on TSP.
    Returns a modified distance matrix that penalizes frequently used edges,
    encouraging the subsequent 2-opt to explore different tour structures.
    
    Uses adaptive local penalties based on edge "regret" (how easy it is to
    break each tour edge), combined with global usage-based penalties.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s limit
    
    n = edge_distance.shape[0]
    
    # Get the edges in the current local optimal tour
    tour = np.asarray(local_opt_tour)
    if len(tour) < 2:
        return edge_distance.copy()
    
    # Check if we have any usage history
    max_usage = edge_n_used.max()
    if max_usage == 0:
        return edge_distance.copy()
    
    # --- Global penalty: smooth saturation based on usage counts ---
    # penalty_global = alpha_global * (edge_n_used / (1 + edge_n_used))
    alpha_global = 0.25
    penalty_global = alpha_global * (edge_n_used / (1.0 + edge_n_used))
    
    # --- Local penalty: adaptive based on edge regret ---
    # For each edge in the current tour, compute how "expensive" it is to break.
    # An edge (u, v) in the tour has neighbors: u's other tour neighbor and v's other tour neighbor.
    # The "regret" of breaking edge (u,v) is approximately:
    #   d(u, v_next) + d(v, u_prev) - d(u, v)
    # where u_prev is u's other neighbor in the tour and v_next is v's other neighbor.
    # Low regret means the edge is easy to break -> penalize more.
    
    penalty_local = np.zeros_like(edge_distance)
    
    # Precompute tour adjacency
    tour_edges = []
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_edges.append((u, v))
    
    # For each edge, compute regret
    # Edge (tour[i], tour[i+1]): 
    #   u = tour[i], v = tour[(i+1) % n]
    #   u's other neighbor: tour[(i-1) % n]
    #   v's other neighbor: tour[(i+2) % n]
    #   regret = d(u, v_other) + d(v, u_other) - d(u, v)
    regrets = np.zeros(n)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        u_other = tour[(i - 1) % n]
        v_other = tour[(i + 2) % n]
        # Regret: cost increase if we swap out this edge
        regret = edge_distance[u, v_other] + edge_distance[v, u_other] - edge_distance[u, v]
        regrets[i] = regret
    
    # Normalize regrets to [0, 1] range for penalty scaling
    # Low regret (easy to break) -> high penalty
    # High regret (hard to break) -> low penalty
    regret_min = np.min(regrets)
    regret_max = np.max(regrets)
    regret_range = regret_max - regret_min
    if regret_range > 1e-10:
        # Invert: low regret gets high penalty
        normalized_regret = (regret_max - regrets) / regret_range  # [0,1], 1 = easy to break
    else:
        normalized_regret = np.ones(n)
    
    # Scale local penalty: base penalty modulated by normalized regret
    alpha_local_base = 0.3
    alpha_local_max = 0.8
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        # Penalty ranges from alpha_local_base (hard to break) to alpha_local_max (easy to break)
        local_pen = alpha_local_base + (alpha_local_max - alpha_local_base) * normalized_regret[i]
        penalty_local[u, v] = local_pen
        penalty_local[v, u] = local_pen
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple global penalty only
        modified_distance = edge_distance * (1.0 + penalty_global)
        modified_distance = (modified_distance + modified_distance.T) / 2.0
        return np.nan_to_num(modified_distance, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Combine penalties
    total_penalty = penalty_global + penalty_local
    
    # Apply penalty to the distance matrix
    # Multiplicative form: d' = d * (1 + penalty)
    modified_distance = edge_distance * (1.0 + total_penalty)
    
    # Ensure symmetry
    modified_distance = (modified_distance + modified_distance.T) / 2.0
    
    # Ensure finite values
    modified_distance = np.nan_to_num(modified_distance, nan=0.0, posinf=1e10, neginf=0.0)
    
    return modified_distance
# EVOLVE-BLOCK-END
