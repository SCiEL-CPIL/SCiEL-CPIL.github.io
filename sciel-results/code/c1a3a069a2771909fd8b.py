"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Repulsive force model: penalty inversely proportional to edge length.
       Short edges get disproportionately large penalties, creating a strong
       gradient to escape local optima by exploring alternative short connections.
    2. Usage-based penalty with non-linear scaling (power 1.5) for more aggressive
       escape from frequently-used edge patterns.
    3. Tour-edge penalty: edges in the current local-opt tour are penalized
       to force 2-opt to break away.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Deadline guard: if we've used too much time, return a safe perturbation
    deadline = start_time + 280.0  # 280s out of 300s total budget
    
    # Compute median edge length for normalization
    # Use only upper triangle to avoid double-counting
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_edges = edge_distance[mask]
    median_dist = np.median(upper_edges)
    if median_dist <= 0:
        median_dist = 1.0
    
    # 1. Repulsive force model: penalty inversely proportional to edge length
    # Short edges get large penalties, long edges get small penalties
    # Use a smooth inverse: penalty = alpha * (median_dist / (edge_distance + epsilon))^beta
    # This creates a strong gradient for short edges
    epsilon = 1e-8
    ratio = median_dist / (edge_distance + epsilon)
    # Clip ratio to avoid extreme values
    ratio_clipped = np.clip(ratio, 0.0, 10.0)
    # Use power 1.5 for stronger repulsion of short edges
    repulsive_penalty = 0.4 * (ratio_clipped ** 1.5 - 1.0)
    # Only apply positive penalties (short edges relative to median)
    repulsive_penalty = np.maximum(0.0, repulsive_penalty)
    
    # 2. Usage-based penalty with non-linear scaling
    # alpha controls the maximum additional penalty factor
    alpha_usage = 0.25
    # Use power 1.5 for more aggressive escape
    usage_penalty = alpha_usage * (edge_n_used ** 1.5) / (1.0 + edge_n_used ** 1.5)
    
    # 3. Tour-edge penalty: force breaking from current tour
    immediate_scale = 0.15
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    immediate_penalty = immediate_scale * tour_mask
    
    # Check deadline
    if time.time() > deadline:
        # Return a minimal but valid perturbation
        result = edge_distance * (1.0 + immediate_penalty)
        result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
        return result
    
    # Combine all penalties
    total_penalty = repulsive_penalty + usage_penalty + immediate_penalty
    
    # Apply multiplicatively to keep all distances positive and finite
    result = edge_distance * (1.0 + total_penalty)
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
