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
    
    # Design new heuristic based on binary MST membership
    # High constant for MST edges, low constant for others
    # This forces the ACO to prioritize the structural backbone
    
    high_val = 1.0
    low_val = 0.1
    
    heuristic = np.full((n, n), low_val, dtype=float)
    heuristic[mst_adj] = high_val
    
    # Ensure diagonal is 0 (or very small)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure no non-positive values (replace with small positive)
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # So we must ensure all off-diagonal are > 0.
    # Our low_val is 0.1 which is > 0.
    # But we should be safe.
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
