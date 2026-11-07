"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_mst_edges(distance_matrix):
    """Compute MST edges using Prim's algorithm."""
    n = distance_matrix.shape[0]
    in_mst = [False] * n
    min_dist = [np.inf] * n
    parent = [-1] * n
    
    min_dist[0] = 0.0
    mst_edges = []
    
    for _ in range(n):
        u = -1
        best_dist = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best_dist:
                best_dist = min_dist[i]
                u = i
        
        if u == -1:
            break
            
        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.append((parent[u], u))
            
        for v in range(n):
            if not in_mst[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                parent[v] = u
                
    return mst_edges

def _greedy_tour(distance_matrix, start_node=0):
    """Generate a greedy nearest-neighbor tour."""
    n = distance_matrix.shape[0]
    visited = [False] * n
    tour = [start_node]
    visited[start_node] = True
    current = start_node
    
    for _ in range(n - 1):
        nearest = -1
        min_d = np.inf
        for v in range(n):
            if not visited[v]:
                d = distance_matrix[current, v]
                if d < min_d:
                    min_d = d
                    nearest = v
        tour.append(nearest)
        visited[nearest] = True
        current = nearest
        
    tour.append(start_node)
    return tour

def _two_opt(tour, distance_matrix, time_limit=5.0):
    """Optimize tour using 2-Opt."""
    start_time = time.time()
    n = len(tour) - 1
    # tour is a list of nodes, tour[0] == tour[-1]
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(1, n - 1):
            if (time.time() - start_time) > time_limit:
                return tour
            for j in range(i + 1, n):
                if (time.time() - start_time) > time_limit:
                    return tour
                
                d1 = distance_matrix[tour[i-1], tour[i]]
                d2 = distance_matrix[tour[j], tour[j+1]]
                d3 = distance_matrix[tour[i-1], tour[j]]
                d4 = distance_matrix[tour[i], tour[j+1]]
                
                if d3 + d4 < d1 + d2 - 1e-10:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
                    
    return tour

def _or_opt(tour, distance_matrix, time_limit=5.0):
    """Optimize tour using Or-Opt (move single city)."""
    start_time = time.time()
    n = len(tour) - 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(1, n):
            if (time.time() - start_time) > time_limit:
                return tour
            city = tour[i]
            # Remove city from position i
            prev_city = tour[i-1]
            next_city = tour[i+1]
            
            # Calculate length change for removing city
            # Old edges: (prev, city), (city, next)
            # New edge: (prev, next)
            old_len = distance_matrix[prev_city, city] + distance_matrix[city, next_city]
            
            # Try inserting city at other positions
            for j in range(1, n):
                if j == i or j == i - 1:
                    continue
                if (time.time() - start_time) > time_limit:
                    return tour
                
                # Position j means inserting between tour[j-1] and tour[j]
                # But we need to be careful with indices after removal
                # Let's just do a simple check: try all insertion points
                # For simplicity and correctness, let's stick to 2-opt for now
                # Or-Opt is complex to implement correctly with the closed tour structure
                # without careful index management.
                pass
                    
    return tour

def heuristics(distance_matrix):
    # Ensure input is a numpy array
    dist = np.asarray(distance_matrix, dtype=float)
    n = dist.shape[0]
    
    # Initialize heuristic with inverse distance
    # Avoid division by zero
    heuristic = 1.0 / (dist + 1e-9)
    
    # 1. Boost MST edges
    mst_edges = _compute_mst_edges(dist)
    for u, v in mst_edges:
        heuristic[u, v] *= 3.0
        heuristic[v, u] *= 3.0
        
    # 2. Boost 2-Opt optimized greedy tour edges
    greedy_tour = _greedy_tour(dist, start_node=0)
    optimized_tour = _two_opt(greedy_tour, dist, time_limit=2.0)
    
    for i in range(len(optimized_tour) - 1):
        u = optimized_tour[i]
        v = optimized_tour[i+1]
        heuristic[u, v] *= 8.0
        heuristic[v, u] *= 8.0
        
    # Ensure positivity and finiteness
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
