# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _nn_tour(points):
    """Nearest neighbor tour construction."""
    n = len(points)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        best = -1
        best_d = float('inf')
        for j in range(n):
            if not visited[j]:
                d = np.linalg.norm(points[last] - points[j])
                if d < best_d:
                    best_d = d
                    best = j
        tour.append(best)
        visited[best] = True
    return tour

def _two_opt(tour, points, max_iters=100):
    """Simple 2-opt improvement."""
    n = len(tour)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                old = np.linalg.norm(points[a] - points[b]) + np.linalg.norm(points[c] - points[d])
                new = np.linalg.norm(points[a] - points[c]) + np.linalg.norm(points[b] - points[d])
                if new < old - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour

def _mst_edges(points):
    """Compute MST edges using Prim's algorithm."""
    n = len(points)
    visited = [False] * n
    dist = [float('inf')] * n
    dist[0] = 0
    parent = [-1] * n
    edges = set()
    
    for _ in range(n):
        # Find unvisited node with min dist
        min_d = float('inf')
        min_node = -1
        for v in range(n):
            if not visited[v] and dist[v] < min_d:
                min_d = dist[v]
                min_node = v
        if min_node == -1:
            break
        visited[min_node] = True
        if parent[min_node] != -1:
            a, b = min_node, parent[min_node]
            a, b = min(a, b), max(a, b)
            edges.add((a, b))
        for v in range(n):
            if not visited[v]:
                d = np.linalg.norm(points[min_node] - points[v])
                if d < dist[v]:
                    dist[v] = d
                    parent[v] = min_node
    return edges

def _knn_edges(points, k=3):
    """Compute k-NN edges."""
    n = len(points)
    edges = set()
    for i in range(n):
        dists = np.linalg.norm(points - points[i], axis=1)
        # Exclude self
        dists[i] = float('inf')
        neighbors = np.argsort(dists)[:k]
        for j in neighbors:
            a, b = i, j
            a, b = min(a, b), max(a, b)
            edges.add((a, b))
    return edges

def _edge_crossing_score(points):
    """Score edges by how many other edges they cross."""
    n = len(points)
    crossing = np.zeros((n, n), dtype=np.int32)
    
    def crosses(p1, p2, p3, p4):
        """Check if segment (p1,p2) crosses (p3,p4)."""
        def orient(a, b, c):
            val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
            if val > 1e-10:
                return 1
            if val < -1e-10:
                return -1
            return 0
        
        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)
        
        if o1 != o2 and o3 != o4:
            return True
        return False
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k == i or k == j:
                    continue
                for l in range(k + 1, n):
                    if l == i or l == j:
                        continue
                    if crosses(points[i], points[j], points[k], points[l]):
                        crossing[i, j] += 1
                        crossing[j, i] += 1
    return crossing

def heuristics(distance_matrix):
    """
    Compute heuristic matrix for ACO TSP.
    Returns a matrix where larger values indicate more attractive edges.
    """
    start_time = time.time()
    deadline = 290  # Leave 10s safety margin
    
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.ones_like(distance_matrix)
    
    # Extract coordinates from distance matrix (approximate using first two columns)
    # For uniform random points in unit square, we can try to recover coordinates
    # Using a simple approach: assume the distance matrix is consistent with 2D points
    # We'll use a heuristic approach to estimate positions
    
    # Simple approach: use the distance matrix directly for some computations
    # For crossing detection, we need coordinates. Let's try to estimate them.
    
    # Use a simple embedding: project based on distances from node 0
    # This is a rough estimate
    try:
        # Estimate 2D coordinates using distances
        # Use node 0 as reference
        d0 = distance_matrix[0, :].copy()
        d0[0] = 0
        
        # Try to get x-coordinates from distances
        # For points in unit square, we can use a simple projection
        # x_i ≈ (d0_i^2 + 1 - d0_i^2) / 2 ... this doesn't work well
        
        # Better: use the fact that for uniform random points, 
        # we can approximate coordinates by using the distance matrix structure
        # Use a simple heuristic: assume x increases with distance from node 0
        # This is rough but should give reasonable results for the heuristic
        
        # Alternative: just use the distance matrix directly without coordinates
        # Skip crossing detection if we can't get good coordinates
        use_crossing = False
        points = None
        
        # Try to recover coordinates using a simple method
        # For 50 points in unit square, we can try
        # x_i = (d0_i^2 + avg_d2 - 1) / (2 * avg_d) ... not reliable
        
        # Just skip crossing detection and use other signals
        # We can still use NN tour, MST, k-NN
        
    except Exception:
        use_crossing = False
        points = None
    
    # Compute NN tour using distance matrix
    def _nn_tour_dm(dm):
        nn = len(dm)
        visited = [False] * nn
        tour = [0]
        visited[0] = True
        for _ in range(nn - 1):
            last = tour[-1]
            best = -1
            best_d = float('inf')
            for j in range(nn):
                if not visited[j]:
                    if dm[last, j] < best_d:
                        best_d = dm[last, j]
                        best = j
            tour.append(best)
            visited[best] = True
        return tour
    
    def _two_opt_dm(tour, dm):
        nn = len(tour)
        improved = True
        iters = 0
        max_iters = 50
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(nn - 1):
                for j in range(i + 2, nn):
                    if i == 0 and j == nn - 1:
                        continue
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % nn]
                    old = dm[a, b] + dm[c, d]
                    new = dm[a, c] + dm[b, d]
                    if new < old - 1e-10:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
        return tour
    
    tour = _nn_tour_dm(distance_matrix)
    tour = _two_opt_dm(tour, distance_matrix)
    
    # Get tour edges
    tour_edges = set()
    for i in range(n):
        a, b = tour[i], tour[(i + 1) % n]
        a, b = min(a, b), max(a, b)
        tour_edges.add((a, b))
    
    # Compute MST edges using Prim's on distance matrix
    def _mst_edges_dm(dm):
        nn = len(dm)
        visited = [False] * nn
        dist = [float('inf')] * nn
        dist[0] = 0
        parent = [-1] * nn
        edges = set()
        
        for _ in range(nn):
            min_d = float('inf')
            min_node = -1
            for v in range(nn):
                if not visited[v] and dist[v] < min_d:
                    min_d = dist[v]
                    min_node = v
            if min_node == -1:
                break
            visited[min_node] = True
            if parent[min_node] != -1:
                a, b = min_node, parent[min_node]
                a, b = min(a, b), max(a, b)
                edges.add((a, b))
            for v in range(nn):
                if not visited[v]:
                    if dm[min_node, v] < dist[v]:
                        dist[v] = dm[min_node, v]
                        parent[v] = min_node
        return edges
    
    mst_e = _mst_edges_dm(distance_matrix)
    
    # Compute k-NN edges
    def _knn_edges_dm(dm, k=3):
        nn = len(dm)
        edges = set()
        for i in range(nn):
            dists = dm[i, :].copy()
            dists[i] = float('inf')
            neighbors = np.argsort(dists)[:k]
            for j in neighbors:
                a, b = i, j
                a, b = min(a, b), max(a, b)
                edges.add((a, b))
        return edges
    
    knn_e = _knn_edges_dm(distance_matrix, k=3)
    
    # Build heuristic
    heuristic = np.ones((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_matrix[i, j]
            
            # Base: inverse distance
            h = 1.0 / (d + 1e-9)
            
            # Boost for tour edges (strongest signal)
            if (i, j) in tour_edges:
                h *= 10.0
            # Boost for MST edges
            if (i, j) in mst_e:
                h *= 3.0
            # Boost for k-NN edges
            if (i, j) in knn_e:
                h *= 2.0
            
            heuristic[i, j] = h
            heuristic[j, i] = h
    
    # Set diagonal to 0 (self-loops not used)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all entries are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return base heuristic as fallback
        base = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(base, 0.0)
        return np.maximum(base, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
