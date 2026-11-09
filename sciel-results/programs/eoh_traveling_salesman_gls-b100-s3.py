"""Guided local search penalty rule for TSP with adaptive edge penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.perf_counter()
_DEADLINE = _start_time + 310.0  # Safety margin before 330s kill

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Adaptive GLS penalty: combines local tour-edge penalties (length-proportional,
    boosted by usage) with a smooth global usage penalty on all edges.
    This creates strong repulsion from the current basin while also discouraging
    frequently-used historical edges, encouraging exploration of new structures.
    """
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()
    
    n = len(local_opt_tour)
    if n == 0:
        return edge_distance.copy()
    
    # Create a copy of the true distance matrix
    penalized_dist = edge_distance.copy()
    
    # Extract edges of the current local optimum tour
    tour = np.asarray(local_opt_tour)
    u = tour
    v = np.roll(tour, -1)
    
    # Get true distances for these tour edges
    current_edge_lengths = edge_distance[u, v]
    
    # Get usage counts for current tour edges
    edge_counts = edge_n_used[u, v]
    
    # --- Global smooth usage penalty ---
    # Penalize ALL edges based on their historical usage frequency.
    # Use a smooth saturating function: count / (1 + count)
    # This avoids discontinuities and provides meaningful scaling.
    max_usage = np.max(edge_n_used) if edge_n_used.size > 0 else 0
    if max_usage > 0:
        # Normalize to [0, 1) range
        normalized_usage = edge_n_used / (1.0 + max_usage)
        # Smooth penalty: scales with distance, saturates for heavily-used edges
        alpha_global = 0.25
        global_penalty = alpha_global * normalized_usage
        # Apply multiplicatively: d' = d * (1 + penalty)
        penalized_dist = penalized_dist * (1.0 + global_penalty)
    
    # --- Local tour-edge penalty ---
    # Stronger, length-proportional penalty on the current tour's edges,
    # boosted by their usage frequency.
    avg_edge_len = np.mean(current_edge_lengths)
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = np.mean(edge_distance)
        if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
            avg_edge_len = 0.1
    
    # Normalize counts for the current tour edges
    if max_usage == 0:
        max_usage = 1
    
    base_frac = 0.20
    usage_boost = 1.2
    
    for k in range(n):
        ui = u[k]
        vi = v[k]
        edge_len = current_edge_lengths[k]
        count = edge_counts[k]
        norm_count = count / max_usage
        # Penalty scales with edge length and usage frequency
        penalty = base_frac * edge_len * (1.0 + usage_boost * norm_count)
        penalized_dist[ui, vi] += penalty
        penalized_dist[vi, ui] += penalty
    
    # Ensure the diagonal is 0
    np.fill_diagonal(penalized_dist, 0.0)
    
    # Ensure symmetry
    penalized_dist = (penalized_dist + penalized_dist.T) / 2.0
    
    # Check for finiteness
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0, posinf=np.max(edge_distance), neginf=0.0)
    
    return penalized_dist
# EVOLVE-BLOCK-END
