"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute heuristic matrix for ACO TSP.
    Combines inverse distance with structural guidance from a 2-opt improved tour.
    """
    n = distance_matrix.shape[0]
    dm = np.maximum(distance_matrix, 1e-9)
    
    # Base heuristic: inverse cubed distance for strong short-edge preference
    heuristic = 1.0 / (dm ** 3)
    
    # Build a good tour using nearest neighbor + 2-opt
    # Nearest Neighbor construction
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if not visited[j]:
                d = distance_matrix[current, j]
                if d < min_dist:
                    min_dist = d
                    next_node = j
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # 2-opt local search to improve the tour
    improved = True
    max_passes = 30
    pass_count = 0
    
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = tour[j]
                d = tour[(j + 1) % n]
                
                len_old = distance_matrix[a, b] + distance_matrix[c, d]
                len_new = distance_matrix[a, c] + distance_matrix[b, d]
                
                if len_new < len_old - 1e-10:
                    # Reverse segment from i+1 to j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
    
    # Boost edges from the improved tour
    boost_factor = 5.0
    for k in range(n):
        u = tour[k]
        v = tour[(k + 1) % n]
        heuristic[u, v] *= boost_factor
        heuristic[v, u] *= boost_factor
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
