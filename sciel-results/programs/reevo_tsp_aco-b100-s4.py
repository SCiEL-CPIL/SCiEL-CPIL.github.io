"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _nearest_neighbor_tour(d, start):
    """Construct a tour using the Nearest Neighbor heuristic starting from 'start'."""
    n = len(d)
    visited = np.zeros(n, dtype=bool)
    tour = np.zeros(n, dtype=int)
    tour[0] = start
    visited[start] = True
    current = start
    for i in range(1, n):
        # Find the nearest unvisited node
        # Mask visited nodes with infinity
        dists = d[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour[i] = next_node
        visited[next_node] = True
        current = next_node
    return tour

def _two_opt(d, tour):
    """Improve a tour using 2-opt local search."""
    n = len(tour)
    improved = True
    # Map tour indices to node indices for easier distance calculation
    # tour is a sequence of nodes: tour[0], tour[1], ..., tour[n-1]
    # We want to minimize sum(d[tour[i], tour[i+1]]) for i=0..n-2 + d[tour[n-1], tour[0]]
    
    while improved:
        improved = False
        for i in range(n - 1):
            if i + 1 >= n:
                break
            a = tour[i]
            b = tour[i + 1]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue # Reversing the whole tour is the same tour
                c = tour[j]
                d_node = tour[j + 1] if j + 1 < n else tour[0]
                
                # Current edges: (a,b) and (c,d_node)
                # New edges: (a,c) and (b,d_node)
                delta = d[a, c] + d[b, d_node] - d[a, b] - d[c, d_node]
                
                if delta < -1e-10:
                    # Reverse the segment from i+1 to j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
    return tour

def _compute_structural_prior(d):
    """
    Compute a heuristic matrix based on the edges of a good tour found by 2-opt.
    This acts as a strong prior for the ACO.
    """
    n = len(d)
    # Try multiple starting nodes for Nearest Neighbor to find a good initial tour
    best_tour = None
    best_len = np.inf
    
    # Use a few random or fixed starts
    starts = [0, 1, 2, n // 2, n - 1]
    for start in starts:
        tour = _nearest_neighbor_tour(d, start)
        tour = _two_opt(d, tour)
        # Calculate length
        length = 0
        for i in range(n):
            length += d[tour[i], tour[(i + 1) % n]]
        if length < best_len:
            best_len = length
            best_tour = tour.copy()
    
    if best_tour is None:
        # Fallback
        return 1.0 / (d + 1e-9)
        
    # Create an edge attractiveness matrix
    # Edges in the best tour get high attractiveness
    # Other edges get low attractiveness
    # We want to guide ants to follow this structure.
    
    # Heuristic matrix H
    # If (i, j) is an edge in best_tour, H[i, j] = high
    # Else H[i, j] = low
    
    # Let's set H[i, j] = 1 / d[i, j] generally, but boost edges in the tour.
    # Or simply: H[i, j] = 10.0 if in tour, else 1.0 / d[i, j]
    
    H = np.zeros((n, n))
    # Base heuristic: inverse distance
    H[:] = 1.0 / (d + 1e-9)
    
    # Boost edges in the best tour
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        # Boost both directions? ACO is usually directed in construction but symmetric TSP.
        # The problem says "travel direction only" for pheromone, but heuristic is symmetric usually.
        # Let's boost both to be safe, or just the forward direction?
        # "Heuristic matrix ... same shape". "probability proportional to pheromone[i, j] * heuristic[i, j]"
        # If TSP is symmetric, d[i,j]==d[j,i].
        # Boosting both helps if the ant might traverse either way.
        H[u, v] = 100.0
        H[v, u] = 100.0
        
    # Ensure positive
    H[H <= 0] = 1e-9
    return H

def heuristics(distance_matrix):
    start_time = time.time()
    d = np.asarray(distance_matrix)
    n = len(d)
    
    # Deadline guard: if we take too long, fall back to simple inverse distance
    # 2-opt on 50 nodes is very fast, but let's be safe.
    if time.time() - start_time > 250:
        return 1.0 / (d + 1e-9)
        
    try:
        H = _compute_structural_prior(d)
    except Exception:
        H = 1.0 / (d + 1e-9)
        
    # Final safety check
    if time.time() - start_time > 290:
        return 1.0 / (d + 1e-9)
        
    return H
# EVOLVE-BLOCK-END
