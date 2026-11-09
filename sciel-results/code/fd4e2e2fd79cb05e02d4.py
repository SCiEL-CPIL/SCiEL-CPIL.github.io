import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Updates the edge distance matrix by adding penalties to edges
    frequently used in previous local optima (Guided Local Search).
    Uses adaptive penalty scaling based on tour edge lengths.
    """
    start_time = time.time()
    deadline = start_time + 2.0  # Safety margin
    
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    # Extract edges of the current local optimum tour
    tour = np.asarray(local_opt_tour)
    
    # Get edge indices
    u = tour
    v = np.roll(tour, -1)
    
    # Get true distances for current tour edges
    current_edge_lengths = edge_distance[u, v]
    tour_len = np.sum(current_edge_lengths)
    avg_edge_len = np.mean(current_edge_lengths)
    
    # Fallback for degenerate cases
    if not np.isfinite(tour_len) or tour_len == 0:
        tour_len = np.sum(edge_distance) / n
        avg_edge_len = tour_len / n
    if not np.isfinite(avg_edge_len) or avg_edge_len == 0:
        avg_edge_len = 0.1
    
    # Adaptive penalty: scale so that the current tour becomes non-optimal
    # under penalized distances. Use margin to ensure escape.
    margin = 0.08  # Slightly reduced margin to preserve landscape structure
    target_penalty_per_edge = (tour_len * margin) / n
    
    # Base penalty from accumulated usage (scaled by average edge length)
    # Use a decaying penalty to avoid over-penalizing edges used in early rounds
    alpha = 0.07
    # Decay historical counts: older usage matters less
    # edge_n_used is accumulated over rounds; apply exponential decay
    # by scaling down the contribution of frequently-used edges
    # Simple approach: use sqrt to dampen the effect of high counts
    decayed_usage = np.sqrt(edge_n_used + 1.0) - 1.0  # dampens high counts
    base_penalty = edge_distance + (alpha * avg_edge_len) * decayed_usage
    
    # Add targeted penalty to current tour edges
    # This makes the current local optimum suboptimal, forcing 2-opt to escape
    # Use a slightly stronger penalty on the most frequently used tour edges
    # to prioritize escaping the most "stuck" parts of the tour
    edge_usage_in_tour = edge_n_used[u, v]
    # Scale penalty: edges used more often in history get slightly more penalty
    usage_factor = 1.0 + 0.3 * (edge_usage_in_tour / max(1.0, np.max(edge_n_used)))
    base_penalty[u, v] += target_penalty_per_edge * usage_factor
    base_penalty[v, u] += target_penalty_per_edge * usage_factor
    
    # Ensure the diagonal is 0
    np.fill_diagonal(base_penalty, 0.0)
    
    # Ensure symmetry
    penalized_dist = (base_penalty + base_penalty.T) / 2.0
    
    # Check for finiteness
    if not np.all(np.isfinite(penalized_dist)):
        penalized_dist = np.nan_to_num(penalized_dist, nan=0.0, posinf=np.max(edge_distance), neginf=0.0)
        
    # Time check
    if time.time() > deadline:
         return edge_distance.copy()

    return penalized_dist
