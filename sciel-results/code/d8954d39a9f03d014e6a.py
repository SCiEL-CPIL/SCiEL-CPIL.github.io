"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty: penalize edges that have been used in previous
    local optima, using a smooth adaptive penalty with edge-length-aware scaling.
    """
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 300s limit
    
    n = len(local_opt_tour)
    max_usage = edge_n_used.max()
    
    if max_usage == 0:
        return edge_distance.copy()
    
    # Normalize usage to [0, 1] range
    norm_usage = edge_n_used / max_usage
    
    # Smooth saturation penalty: 1 - exp(-beta * norm_usage)
    # This gives gentle penalty for lightly-used edges, stronger for heavily-used
    beta = 2.0  # saturation rate
    
    base_penalty = 1.0 - np.exp(-beta * norm_usage)
    
    # Edge-length-aware scaling: shorter edges are more important to preserve
    # so they should get less penalty
    diag_mask = np.eye(n, dtype=bool)
    
    # Normalize edge lengths to [0, 1] for scaling
    non_diag = ~diag_mask
    min_len = edge_distance[non_diag].min()
    max_len = edge_distance[non_diag].max()
    range_len = max_len - min_len
    
    if range_len > 0:
        norm_lengths = (edge_distance - min_len) / range_len
    else:
        norm_lengths = np.zeros_like(edge_distance)
    
    # Conservative length scaling: longer edges get slightly more penalty
    delta_len = 0.3  # maximum additional penalty multiplier for longest edges
    
    penalty_multiplier = 1.0 + delta_len * norm_lengths
    
    # Adaptive alpha: scale with number of distinct edges in current tour
    # More distinct edges = more spread out = can afford larger penalty
    tour_edges = set()
    for i in range(n):
        j = local_opt_tour[(i + 1) % n]
        edge = (min(i, j), max(i, j))
        tour_edges.add(edge)
    
    # Scale alpha based on how many tour edges have been used before
    used_in_tour = sum(1 for e in tour_edges if edge_n_used[e[0], e[1]] > 0)
    frac_used = used_in_tour / len(tour_edges) if tour_edges else 0
    
    # Start with moderate penalty, increase if many tour edges are stale
    alpha = 0.15 + 0.15 * frac_used
    
    penalty_factor = 1.0 + alpha * base_penalty * penalty_multiplier
    
    # Ensure the matrix remains finite and same shape
    result = edge_distance * penalty_factor
    
    # Safety: ensure no NaN/inf
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=edge_distance)
    
    # Time guard: if we're getting close to deadline, return immediately
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
