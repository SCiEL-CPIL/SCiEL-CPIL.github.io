"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO on TSP.
    
    Strategy:
    1. Base: inverse distance
    2. Boost edges that appear in a nearest-neighbor heuristic tour
    3. Boost edges that are in the MST (minimum spanning tree)
    """
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)
    
    # Base: inverse distance
    base = 1.0 / (distance_matrix + 1e-9)
    np.fill_diagonal(base, 0.0)
    
    # Compute nearest-neighbor tour from node 0
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    for _ in range(n - 1):
        # Find nearest unvisited
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    tour.append(0)  # close the tour
    
    # Mark edges in NN tour
    nn_tour_edges = np.zeros_like(base)
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i + 1]
        nn_tour_edges[u, v] = 1.0
        nn_tour_edges[v, u] = 1.0
    
    # Compute MST using Prim's algorithm
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    mst_edges = np.zeros_like(base)
    for _ in range(n - 1):
        # Find min edge from in_mst to not_in_mst
        best = np.inf
        best_u, best_v = -1, -1
        for u in range(n):
            if not in_mst[u]:
                continue
            for v in range(n):
                if in_mst[v]:
                    continue
                if distance_matrix[u, v] < best:
                    best = distance_matrix[u, v]
                    best_u, best_v = u, v
        in_mst[best_v] = True
        mst_edges[best_u, best_v] = 1.0
        mst_edges[best_v, best_u] = 1.0
    
    # Combine
    heuristic = base * (1.0 + 0.5 * nn_tour_edges + 0.3 * mst_edges)
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
