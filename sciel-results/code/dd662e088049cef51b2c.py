"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturb the edge distance matrix to encourage the 2-opt solver to escape
    the current local optimum.
    
    Strategy:
    1. Calculate the current tour length and max distance.
    2. Identify edges in the current tour and their usage counts.
    3. Apply a strong additive penalty to each tour edge, scaled by usage count
       and a base penalty that guarantees 2-opt will remove these edges.
    4. Return the modified distance matrix.
    """
    n = len(local_opt_tour)
    if n <= 1:
        return edge_distance.copy()
    
    # 1. Calculate current tour length and max distance
    tour_len = 0.0
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_len += edge_distance[u, v]
    
    if tour_len == 0:
        return edge_distance.copy()
    
    max_dist = float(np.max(edge_distance))
    if max_dist == 0 or not np.isfinite(max_dist):
        return edge_distance.copy()
    
    # 2. Identify edges and their usage counts
    edge_usages = []
    edge_indices = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        usage = float(edge_n_used[u, v])
        edge_usages.append(usage)
        edge_indices.append((u, v))
    
    # 3. Calculate penalties
    # Base penalty: large enough to make 2-opt remove the edge
    # Scale by (1 + usage) to emphasize frequently used edges
    # The base is a fraction of max_dist to ensure it dominates
    base_penalty = 0.1 * max_dist
    
    # 4. Create modified distance matrix
    modified_distance = edge_distance.copy()
    
    for i, (u, v) in enumerate(edge_indices):
        usage = edge_usages[i]
        # Penalty increases with usage count
        p = base_penalty * (1.0 + usage)
        # Ensure penalty is positive and finite
        if p > 0 and np.isfinite(p):
            modified_distance[u, v] += p
            modified_distance[v, u] += p
            
    return modified_distance
# EVOLVE-BLOCK-END
