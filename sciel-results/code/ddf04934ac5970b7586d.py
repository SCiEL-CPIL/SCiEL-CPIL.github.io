import numpy as np
import time

_START_TIME = time.monotonic()
_DEADLINE = 290.0  # 10 seconds buffer before the 300s limit (or 330s kill limit)

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Returns a modified distance matrix for the evaluator's 2-opt.
    
    Strategy:
    1. Estimate the current round number from the max value in edge_n_used.
    2. Calculate an adaptive penalty strength that starts high and decays.
    3. Apply a penalty to all edges proportional to their usage count.
    4. Apply a small reward (negative offset) to edges not in the current tour 
       that have low usage counts, to encourage exploration of under-explored motifs.
    """
    # Time guard
    if time.monotonic() - _START_TIME > _DEADLINE:
        return edge_distance.copy()

    # Estimate round number
    max_usage = np.max(edge_n_used)
    if max_usage == 0:
        max_usage = 1.0
    else:
        max_usage = float(max_usage)

    # Adaptive parameters
    # Alpha: Penalty strength. Starts around 0.5, decays.
    # Beta: Reward strength. Small constant or decaying.
    alpha = 0.5 / (1.0 + 0.1 * max_usage)
    beta = 0.05 / (1.0 + 0.05 * max_usage)

    # Create a copy of the distance matrix
    modified_distance = edge_distance.copy()

    # Apply standard GLS penalty to all edges: D' = D + alpha * n_used
    modified_distance += alpha * edge_n_used

    # Identify edges in the current local optimal tour
    n = len(local_opt_tour)
    # Create a mask for edges in the current tour
    current_tour_mask = np.zeros_like(edge_distance, dtype=bool)
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        current_tour_mask[u, v] = True
        current_tour_mask[v, u] = True

    # Apply reward to non-current edges with low usage
    # Condition: not in current tour AND usage < max_usage (or some threshold)
    # Let's use usage < max_usage / 2 to target "fresh" edges
    threshold = max_usage * 0.5
    reward_mask = (~current_tour_mask) & (edge_n_used < threshold)
    
    # Subtract beta from these edges
    modified_distance[reward_mask] -= beta

    # Ensure no negative distances (though unlikely with small beta)
    # And ensure finiteness
    modified_distance = np.maximum(modified_distance, 0.0)
    modified_distance = np.nan_to_num(modified_distance, nan=0.0, posinf=1e6, neginf=0.0)

    return modified_distance
