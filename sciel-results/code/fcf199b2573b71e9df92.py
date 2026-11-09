"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO on TSP.
    
    Strategy:
    1. Base heuristic: inverse distance (standard).
    2. Boost edges that are in the Minimum Spanning Tree (MST) to encourage
       connectivity without cycles initially.
    3. Boost edges that are in the Delaunay triangulation (computed via
       scipy if available, otherwise approximate with k-nearest neighbors)
       to favor local structure.
    4. Apply a smooth penalty for very long edges to discourage crossings.
    5. Normalize and ensure positivity.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n == 0:
        return d.copy()

    # Ensure symmetry and handle diagonal
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)

    eps = 1e-9

    # 1. Base heuristic: inverse distance
    # Use a smoothed inverse to avoid extreme values
    base = 1.0 / (d + eps)
    
    # 2. MST boost
    # Compute MST using Prim's algorithm (simple, no external deps needed for small n)
    # For n=50, this is fast
    mst_mask = np.zeros((n, n), dtype=bool)
    in_mst = [False] * n
    in_mst[0] = True
    min_dist = d[0].copy()
    
    for _ in range(n - 1):
        # Find unvisited node with min distance
        unvisited = [i for i in range(n) if not in_mst[i]]
        if not unvisited:
            break
        # Find min in min_dist among unvisited
        min_val = float('inf')
        min_idx = -1
        for i in unvisited:
            if min_dist[i] < min_val:
                min_val = min_dist[i]
                min_idx = i
        if min_idx == -1:
            break
        in_mst[min_idx] = True
        # Update min_dist
        for i in range(n):
            if not in_mst[i] and d[min_idx, i] < min_dist[i]:
                min_dist[i] = d[min_idx, i]
        # We need to track which edge was used. 
        # Actually, the above Prim's doesn't easily track the parent edge in this simple form.
        # Let's use a proper parent tracking.
        
    # Re-do MST with parent tracking
    in_mst = [False] * n
    in_mst[0] = True
    parent = [-1] * n
    min_dist = d[0].copy()
    
    for _ in range(n - 1):
        unvisited = [i for i in range(n) if not in_mst[i]]
        min_val = float('inf')
        min_idx = -1
        for i in unvisited:
            if min_dist[i] < min_val:
                min_val = min_dist[i]
                min_idx = i
        if min_idx == -1:
            break
        in_mst[min_idx] = True
        parent[min_idx] = np.argmin(np.where(in_mst, d[min_idx, :], d[min_idx, :]))
        # Actually, we need the parent that gave the min_dist.
        # min_dist[i] is the min distance from i to any node in MST.
        # The node that provided it is not directly stored.
        # Let's just mark edges in MST by checking if d[i,j] is the min connection.
        # Simpler: use scipy if available, else skip MST or use a simpler heuristic.
        
    # Let's try to use scipy for MST if available, as it's more robust.
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.sparse import csr_matrix
        # Convert dense to sparse for MST
        # For complete graph, we can pass the dense matrix directly? 
        # minimum_spanning_tree accepts a graph. Let's pass the dense matrix.
        mst = minimum_spanning_tree(d)
        mst_coo = mst.tocoo()
        mst_mask = np.zeros((n, n), dtype=bool)
        for i, j, v in zip(mst_coo.row, mst_coo.col, mst_coo.data):
            mst_mask[i, j] = True
            mst_mask[j, i] = True
    except Exception:
        # Fallback: use k-nearest neighbors as a proxy for local structure
        mst_mask = np.zeros((n, n), dtype=bool)
        k = 4
        for i in range(n):
            # Get k nearest neighbors
            dists = d[i].copy()
            dists[i] = np.inf
            nn = np.argsort(dists)[:k]
            for j in nn:
                mst_mask[i, j] = True
                mst_mask[j, i] = True

    # 3. Delaunay triangulation boost (approximated by k-NN if scipy not available)
    # For n=50, we can use a simple k-NN approach with larger k to approximate Delaunay
    delaunay_mask = np.zeros((n, n), dtype=bool)
    k_delaunay = 8  # Typically Delaunay neighbors for 2D points are ~6
    for i in range(n):
        dists = d[i].copy()
        dists[i] = np.inf
        nn = np.argsort(dists)[:k_delaunay]
        for j in nn:
            delaunay_mask[i, j] = True
            delaunay_mask[j, i] = True

    # 4. Combine heuristics
    # Boost MST edges more strongly
    heuristic = base.copy()
    heuristic[mst_mask] *= 1.5
    heuristic[delaunay_mask] *= 1.2
    
    # 5. Penalize long edges
    # Find the median distance
    dists_flat = d[np.triu_indices(n, k=1)]
    if len(dists_flat) > 0:
        median_dist = np.median(dists_flat)
        # Apply exponential decay for edges longer than median
        long_edge_penalty = np.exp(-d / (median_dist * 1.5))
        heuristic *= long_edge_penalty

    # 6. Ensure positivity and finite values
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # 7. Normalize to avoid extreme values
    # Scale so that max is around 10
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val * 10.0

    return heuristic

# EVOLVE-BLOCK-END
