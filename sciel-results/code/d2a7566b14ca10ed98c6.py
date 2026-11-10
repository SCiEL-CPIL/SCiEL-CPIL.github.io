"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _two_opt(tour, dm):
    """
    Improve a tour using 2-opt local search.
    Returns the improved tour as a list of node indices.
    """
    n = len(tour)
    tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[(j+1)%n]
                
                # Calculate delta: (d(a,c) + d(b,d)) - (d(a,b) + d(c,d))
                # If negative, the swap improves the tour
                current_cost = dm[a, b] + dm[c, d]
                new_cost = dm[a, c] + dm[b, d]
                
                if new_cost < current_cost - 1e-10:
                    # Reverse the segment from i+1 to j
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
    return tour

def _nearest_neighbor_tour(dm):
    """
    Generate an initial tour using the nearest neighbor heuristic starting from node 0.
    """
    n = dm.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    curr = 0
    
    for _ in range(n - 1):
        # Get distances from current node to all unvisited nodes
        dists = dm[curr].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        curr = next_node
        
    return tour

def heuristics(distance_matrix):
    """
    Compute a heuristic matrix that strongly favors edges in a known-good tour
    while penalizing long-range edges that are not part of that tour.
    """
    n = distance_matrix.shape[0]
    
    # 1. Generate a high-quality tour using Nearest Neighbor + 2-opt
    # This is fast for n=50 and provides a strong baseline.
    nn_tour = _nearest_neighbor_tour(distance_matrix)
    optimal_tour = _two_opt(nn_tour, distance_matrix)
    
    # 2. Identify edges in this optimal tour
    # We want to boost these edges significantly.
    edge_in_tour = np.zeros((n, n), dtype=bool)
    for i in range(n):
        u = optimal_tour[i]
        v = optimal_tour[(i + 1) % n]
        edge_in_tour[u, v] = True
        edge_in_tour[v, u] = True
        
    # 3. Construct the heuristic matrix
    # Base heuristic: Inverse square of distance.
    # This strongly penalizes long edges, which is crucial for 2D TSP to avoid crossings.
    # We use distance_matrix directly. Note: distance_matrix[i,i] is 0, so we add epsilon.
    safe_dm = distance_matrix + 1e-9
    base_heuristic = 1.0 / (safe_dm ** 2)
    
    # For edges in the optimal tour, we want them to be extremely attractive.
    # We can simply set them to a large constant or multiply the base heuristic by a large factor.
    # However, if we set them to a constant, we lose the distance information for other edges.
    # Better: Multiply the base heuristic by a large factor for tour edges.
    # Let's try a factor of 100. This makes tour edges 100x more attractive than equivalent non-tour edges.
    # Since tour edges are usually short, their base_heuristic is already high.
    # This reinforces the "good" structure found by 2-opt.
    
    heuristic = base_heuristic.copy()
    heuristic[edge_in_tour] *= 100.0
    
    # Ensure no zeros or negative values (task says non-positive become 1e-9, but we should avoid 0s in input if possible)
    # The diagonal is 0 in distance_matrix, so base_heuristic diagonal is 1e18. 
    # Ants don't pick visited nodes, so diagonal values don't matter much, but let's be safe.
    # Actually, the problem states "Heuristic entries that are not positive become 1e-9".
    # Our heuristic is all positive.
    
    # One more check: if the 2-opt tour was bad, the heuristic might mislead.
    # But 2-opt from NN is generally very good for 50 points.
    
    return heuristic

# EVOLVE-BLOCK-END
