import numpy as np
import time

def _nearest_neighbor_tour(dist, n):
    """Build a nearest neighbor tour starting from node 0."""
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    current = 0
    for _ in range(n - 1):
        # Find nearest unvisited
        unvisited = np.where(~visited)[0]
        if len(unvisited) == 0:
            break
        dists = dist[current, unvisited]
        idx = np.argmin(dists)
        next_node = unvisited[idx]
        visited[next_node] = True
        tour.append(next_node)
        current = next_node
    # Close the tour
    tour.append(0)
    return np.array(tour)

def _two_opt_improve(dist, tour, n, max_iters=200):
    """Improve tour using 2-opt. tour is a closed tour (first==last)."""
    # Extract the path (remove last element which duplicates first)
    path = list(tour[:-1])
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 2):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate improvement
                a, b = path[i], path[i + 1]
                c, d = path[j], path[(j + 1) % n]
                old_cost = dist[a, b] + dist[c, d]
                new_cost = dist[a, c] + dist[b, d]
                if new_cost < old_cost - 1e-10:
                    # Reverse segment i+1 to j
                    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                    improved = True
    return np.array(path + [path[0]])

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Uses structural prior from NN+2-opt tour combined with inverse distance.
    """
    deadline = time.time() + 280  # Safety margin before 300s limit
    n = distance_matrix.shape[0]
    
    # Start with inverse distance as baseline
    heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Build structural prior: NN tour + 2-opt improvement
    nn_tour = _nearest_neighbor_tour(distance_matrix, n)
    
    # Check time
    if time.time() > deadline:
        return heuristic
    
    improved_tour = _two_opt_improve(distance_matrix, nn_tour, n, max_iters=100)
    
    if time.time() > deadline:
        return heuristic
    
    # Create edge set from the improved tour
    # The tour is closed: improved_tour[i] -> improved_tour[i+1] for i in 0..n-1, with improved_tour[n]=improved_tour[0]
    tour_edges = set()
    for i in range(n):
        u = improved_tour[i]
        v = improved_tour[i + 1]
        tour_edges.add((u, v))
        tour_edges.add((v, u))
    
    # Build a strong bias: edges in the good tour get a large additive boost
    # Use a multiplicative enhancement on top of inverse distance
    # For tour edges: boost by factor; for others: keep baseline
    boost_factor = 5.0
    baseline_floor = 1e-9
    
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristic[i, j] = baseline_floor
            elif (i, j) in tour_edges:
                # Strong boost for tour edges
                heuristic[i, j] = max(heuristic[i, j], 1.0) * boost_factor
            else:
                # Keep inverse distance but ensure positive
                heuristic[i, j] = max(heuristic[i, j], baseline_floor)
    
    # Ensure no non-positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
