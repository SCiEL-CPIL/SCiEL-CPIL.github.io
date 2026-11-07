# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.ones_like(distance_matrix, dtype=float)
    
    # Compute MST using Prim's algorithm with proper edge tracking
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_edge_dist = distance_matrix[0].copy()
    min_edge_from = np.zeros(n, dtype=int)
    
    mst_adj = np.zeros((n, n), dtype=bool)
    
    for _ in range(n - 1):
        min_edge_dist_masked = min_edge_dist.copy()
        min_edge_dist_masked[in_mst] = np.inf
        u = np.argmin(min_edge_dist_masked)
        
        in_mst[u] = True
        v = min_edge_from[u]
        mst_adj[u, v] = True
        mst_adj[v, u] = True
        
        for w in range(n):
            if not in_mst[w] and distance_matrix[u, w] < min_edge_dist[w]:
                min_edge_dist[w] = distance_matrix[u, w]
                min_edge_from[w] = u
    
    # Compute degree of each node in the MST
    mst_degree = mst_adj.sum(axis=1)
    
    # Create a base heuristic based on inverse distance
    # Use a power to make closer edges more attractive
    # Avoid division by zero on diagonal
    dist_safe = distance_matrix.copy()
    np.fill_diagonal(dist_safe, np.inf)
    
    # Inverse distance heuristic: 1/d^alpha
    # Use alpha=1 for standard inverse distance
    alpha = 1.0
    inv_dist = 1.0 / (dist_safe ** alpha)
    
    # Boost MST edges significantly
    # The idea: MST edges get a high multiplier, non-MST edges get lower
    # This creates a sparse guidance structure
    
    # Set heuristic: combine inverse distance with MST boost
    heuristic = inv_dist.copy()
    
    # For MST edges, apply a strong boost
    mst_boost = 5.0
    heuristic[mst_adj] *= mst_boost
    
    # For non-MST edges, reduce slightly to emphasize MST structure
    non_mst_mask = ~mst_adj
    non_mst_mask = non_mst_mask & (distance_matrix > 0)  # exclude diagonal
    heuristic[non_mst_mask] *= 0.3
    
    # Ensure diagonal is 0 (or very small)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure no non-positive values (replace with small positive)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
