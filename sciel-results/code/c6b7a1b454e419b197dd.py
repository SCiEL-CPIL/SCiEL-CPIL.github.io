"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    
    # Helper to calculate tour length
    def get_tour_length(t):
        # Vectorized length calculation
        # t is a list of node indices
        # Length = sum(dist[t[i], t[i+1]] for i in range(n))
        # Using numpy for speed
        idx = np.array(t)
        next_idx = np.roll(idx, -1)
        return np.sum(distance_matrix[idx, next_idx])

    # 2-Opt improvement
    def two_opt(t):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a = t[i]
                    b = t[i + 1]
                    c = t[j]
                    d = t[(j + 1) % n]
                    
                    # Check if reversing segment (i+1, ..., j) improves length
                    # Old edges: (a, b) and (c, d)
                    # New edges: (a, c) and (b, d)
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
                    if delta < 0:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
        return t

    # Nearest Neighbor construction
    def nearest_neighbor(start_node):
        visited = np.zeros(n, dtype=bool)
        visited[start_node] = True
        tour = [start_node]
        current = start_node
        
        for _ in range(n - 1):
            # Find nearest unvisited node
            # Mask visited nodes with inf
            dists = distance_matrix[current, :].copy()
            dists[visited] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        return tour

    # Main search loop: Multiple restarts
    best_tour = None
    best_len = float('inf')
    
    # Try a few random starts. For N=50, 20-50 restarts is fast.
    num_restarts = 50
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        start_node = np.random.randint(n)
        tour = nearest_neighbor(start_node)
        tour = two_opt(tour)
        length = get_tour_length(tour)
        
        if length < best_len:
            best_len = length
            best_tour = tour
    
    # If no tour was found (should not happen), fallback to NN from 0
    if best_tour is None:
        best_tour = nearest_neighbor(0)
        best_tour = two_opt(best_tour)

    # Build Heuristic Matrix
    # Base: Inverse distance
    diag_mask = np.eye(n, dtype=bool)
    safe_dist = distance_matrix.copy()
    safe_dist[diag_mask] = 1.0 # Avoid div by zero
    
    base = 1.0 / safe_dist
    
    # Boost edges in the best reference tour
    # We want the ACO to strongly prefer these edges.
    # A factor of 1000 is generally effective.
    for i in range(n):
        j = (i + 1) % n
        base[best_tour[i], best_tour[j]] = 1000.0
        
    # Ensure diagonal is not attractive
    base[diag_mask] = 1e-9
    
    # Ensure all values are positive and finite
    base = np.maximum(base, 1e-9)
    base = np.where(np.isfinite(base), base, 1e-9)
    
    return base
# EVOLVE-BLOCK-END
