"""Guided local search penalty rule for TSP with adaptive penalty."""
import numpy as np
import time

# Track wall-clock time for safety
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Strategy:
    - Compute a dynamic penalty based on the tour's average edge length and round count.
    - Apply additive penalty to edges in the current local optimum, weighted by usage.
    - Use a smooth, adaptive penalty that increases with both usage and round number.
    - Ensure finite output.
    """
    # Check time limit; if exceeded, return original distances to avoid timeout
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Get edges of the current local optimal tour
    tour = local_opt_tour
    if len(tour) < 2:
        return edge_distance.copy()
    
    # Extract tour edges
    tour_edges = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
    
    # Compute average edge length of current tour for scale normalization
    tour_lengths = [edge_distance[u, v] for u, v in tour_edges]
    avg_tour_edge_len = np.mean(tour_lengths) if tour_lengths else 1.0
    
    # If avg_tour_edge_len is too small, use a default
    if avg_tour_edge_len < 1e-6:
        avg_tour_edge_len = 1.0
    
    # Get max usage count for normalization and infer round number
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used == 0:
        max_used = 1
    
    # Infer the current round from max usage (each round adds 1 to used edges)
    # This helps ramp up the penalty over time
    round_num = float(max_used)
    
    # Dynamic penalty: ramps up with usage and round number
    # Base penalty coefficient scaled by tour edge length
    # We want the penalty to be significant enough to change the 2-opt landscape
    # but not so large that it destroys all useful structure
    
    # Penalty for each edge in the tour: proportional to usage count
    # Use a smooth, adaptive scaling
    # The penalty grows with round number to encourage more aggressive exploration later
    base_penalty = 0.3 * avg_tour_edge_len * (1.0 + 0.1 * round_num)
    
    # Create penalty matrix
    penalty = np.zeros_like(edge_distance, dtype=np.float64)
    for u, v in tour_edges:
        # Weight by usage: more used edges get more penalty
        usage = edge_n_used[u, v]
        # Scale usage to [0, 1] range, then apply penalty
        normalized_usage = usage / (1.0 + max_used)
        # Penalty increases with usage, using a smooth function
        edge_penalty = base_penalty * (0.5 + normalized_usage)
        penalty[u, v] = edge_penalty
        penalty[v, u] = edge_penalty
    
    # Also penalize edges that have been used frequently even if not in current tour
    # This helps break out of local optima by discouraging repeatedly used edges
    # But keep this penalty smaller than the tour edge penalty
    global_penalty = 0.05 * avg_tour_edge_len * (1.0 + 0.05 * round_num)
    usage_normalized = edge_n_used / (1.0 + max_used)
    # Only penalize edges with high usage, not in current tour
    mask = np.ones_like(edge_distance, dtype=bool)
    for u, v in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    global_penalty_matrix = np.where(mask, global_penalty * usage_normalized, 0.0)
    
    # Combine penalties
    total_penalty = penalty + global_penalty_matrix
    
    # Apply penalty to edge distances
    result = edge_distance + total_penalty
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=np.finfo(np.float64).max / 10, neginf=0.0)
    
    # Ensure symmetry
    result = (result + result.T) / 2.0
    
    # Ensure diagonal is zero (distance from a node to itself)
    np.fill_diagonal(result, 0.0)
    
    return result
