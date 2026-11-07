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
       to force 2-opt to break away from the current solution.
    4. All penalties are applied multiplicatively to keep distances positive.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Deadline guard
    deadline = start_time + 280.0
    
    # Compute median edge length for normalization
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_edges = edge_distance[mask]
    median_dist = np.median(upper_edges)
    if median_dist <= 0:
        median_dist = 1.0
    
    # 1. Repulsive force model: penalty ~ 1/d for short edges
    # Use a smooth inverse-distance penalty that's bounded
    # penalty = alpha_geo * (median_dist / (edge_dist + eps)) for edge_dist < median_dist
    eps = 1e-10
    # Only apply to edges shorter than median to avoid penalizing long edges
    short_mask = edge_distance < median_dist
    # Inverse distance scaling: stronger penalty for shorter edges
    inv_dist = np.where(short_mask, median_dist / (edge_distance + eps), 1.0)
    # Clip to avoid extreme values
    inv_dist = np.clip(inv_dist, 1.0, 5.0)
    geometric_penalty = 0.4 * (inv_dist - 1.0) * short_mask
    
    # 2. Usage-based penalty with power-law scaling
    # More aggressive escape from frequently-used edges
    alpha_usage = 0.3
    # Power 1.5 gives stronger penalty for high-usage edges
    usage_penalty = alpha_usage * np.power(edge_n_used, 1.5) / (1.0 + np.power(edge_n_used, 1.5))
    
    # 3. Tour-edge penalty: force breaking from current tour
    immediate_scale = 0.2
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    immediate_penalty = immediate_scale * tour_mask
    
    # Check deadline
    if time.time() > deadline:
        result = edge_distance * (1.0 + immediate_penalty)
        result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
        return result
    
    # Combine all penalties
    total_penalty = geometric_penalty + usage_penalty + immediate_penalty
    
    # Apply multiplicatively to keep all distances positive and finite
    result = edge_distance * (1.0 + total_penalty)
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
