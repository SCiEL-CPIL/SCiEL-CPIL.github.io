"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Compute heuristic matrix for ACO on TSP.
    Uses MST-based structural guidance combined with inverse distance.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    dm = distance_matrix + 1e-9
    
    # 1. Base heuristic: inverse distance with exponent 1.5
    heuristic = 1.0 / (dm ** 1.5)
    
    # 2. Compute Minimum Spanning Tree (MST) for structural guidance
    # Use Prim's algorithm starting from node 0
    in_mst = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0.0
    parent = [-1] * n
    
    for _ in range(n):
        # Find the vertex with minimum distance not yet in MST
        u = -1
        best = float('inf')
        for v in range(n):
            if not in_mst[v] and min_dist[v] < best:
                best = min_dist[v]
                u = v
        
        if u == -1:
            break
        
        in_mst[u] = True
        
        # Update distances for neighbors
        for v in range(n):
            if not in_mst[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                parent[v] = u
    
    # 3. Build MST edge set
    mst_edges = set()
    for v in range(1, n):
        if parent[v] != -1:
            u, v_min = parent[v], v
            mst_edges.add((min(u, v_min), max(u, v_min)))
    
    # 4. Boost MST edges significantly
    mst_boost = 3.0
    for (u, v) in mst_edges:
        heuristic[u, v] *= mst_boost
        heuristic[v, u] *= mst_boost
    
    # 5. Run a greedy + 2-opt local search to find a good tour
    unvisited = set(range(1, n))
    current = 0
    tour = [0]
    
    for _ in range(n - 1):
        # Find nearest unvisited neighbor
        min_d = float('inf')
        best_next = -1
        for node in unvisited:
            if distance_matrix[current, node] < min_d:
                min_d = distance_matrix[current, node]
                best_next = node
        tour.append(best_next)
        unvisited.remove(best_next)
        current = best_next
    
    def tour_length(tour):
        length = 0.0
        for i in range(len(tour)):
            length += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return length
    
    # 2-Opt local search
    improved_tour = tour[:]
    best_length = tour_length(improved_tour)
    improved = True
    max_iterations = 50
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                a = improved_tour[i - 1]
                b = improved_tour[i]
                c = improved_tour[j]
                d = improved_tour[(j + 1) % n]
                
                old_dist = distance_matrix[a, b] + distance_matrix[c, d]
                new_dist = distance_matrix[a, c] + distance_matrix[b, d]
                
                if new_dist < old_dist - 1e-10:
                    improved_tour = improved_tour[:i] + improved_tour[i:j+1][::-1] + improved_tour[j+1:]
                    best_length = new_dist + (best_length - old_dist)
                    improved = True
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
    
    # 6. Boost edges in the improved tour
    tour_boost = 2.0
    for i in range(n):
        u = improved_tour[i]
        v = improved_tour[(i + 1) % n]
        heuristic[u, v] *= tour_boost
        heuristic[v, u] *= tour_boost
    
    # 7. Add k-nearest neighbor boost for broader guidance
    k = 3
    for i in range(n):
        # Get k nearest neighbors
        sorted_nodes = np.argsort(distance_matrix[i])
        for j in sorted_nodes[:k]:
            if j != i:
                heuristic[i, j] *= 1.15
                heuristic[j, i] *= 1.15
    
    # 8. Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # 9. Time check
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # If we exceeded time, return the base heuristic with MST boost
        base = 1.0 / (dm ** 1.5)
        for (u, v) in mst_edges:
            base[u, v] *= mst_boost
            base[v, u] *= mst_boost
        return np.where(np.isfinite(base) & (base > 0), base, 1e-9)
        
    return heuristic
# EVOLVE-BLOCK-END
