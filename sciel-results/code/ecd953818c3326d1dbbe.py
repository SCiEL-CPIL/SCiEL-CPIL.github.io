import numpy as np

def heuristics(distance_matrix):
    d = np.array(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Compute nearest-neighbor tour starting from node 0
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        # Find nearest unvisited node
        unvisited_mask = ~visited
        unvisited_mask[current] = True  # exclude self
        # Get distances from current to all unvisited nodes
        dists_to_unvisited = d[current].copy()
        dists_to_unvisited[~unvisited_mask] = np.inf
        next_node = np.argmin(dists_to_unvisited)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # Build topological heuristic: edges in the NN tour get high attractiveness
    # This provides global structural guidance
    topo_heuristic = np.ones((n, n), dtype=np.float64)
    
    for i in range(n):
        j = tour[(i + 1) % n]
        k = tour[(i - 1) % n]
        # Both directions of tour edges get boosted
        topo_heuristic[i, j] = 10.0
        topo_heuristic[j, i] = 10.0
        topo_heuristic[i, k] = 10.0
        topo_heuristic[k, i] = 10.0
    
    # Also add 2-opt improvement to get a better tour structure
    improved_tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate change in tour length if we reverse segment i+1 to j
                old_len = d[improved_tour[i], improved_tour[i+1]] + d[improved_tour[j], improved_tour[(j+1)%n]]
                new_len = d[improved_tour[i], improved_tour[j]] + d[improved_tour[i+1], improved_tour[(j+1)%n]]
                if new_len < old_len - 1e-10:
                    improved_tour[i+1:j+1] = reversed(improved_tour[i+1:j+1])
                    improved = True
    
    # Rebuild topological heuristic from improved tour
    topo_heuristic = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        j = improved_tour[(i + 1) % n]
        k = improved_tour[(i - 1) % n]
        topo_heuristic[i, j] = 10.0
        topo_heuristic[j, i] = 10.0
        topo_heuristic[i, k] = 10.0
        topo_heuristic[k, i] = 10.0
    
    # Combine with geometric heuristic: inverse distance
    # Find minimum non-zero distance for normalization
    mask = ~np.eye(n, dtype=bool)
    non_zero_dists = d[mask]
    non_zero_dists = non_zero_dists[non_zero_dists > 1e-10]
    
    if len(non_zero_dists) == 0:
        return topo_heuristic
    
    d_min = np.min(non_zero_dists)
    d_safe = np.where(d < 1e-10, d_min, d)
    
    # Geometric heuristic: normalized inverse distance
    geo_heuristic = (d_min / d_safe) ** 1.5
    
    # Ensure no NaNs or Infs
    geo_heuristic = np.nan_to_num(geo_heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Combine: product of topological and geometric heuristics
    # This strongly reinforces edges that are both in the good tour AND geometrically short
    eta = topo_heuristic * geo_heuristic
    
    # Ensure diagonal is small (not used in ACO, but for safety)
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure no NaNs or Infs
    eta = np.nan_to_num(eta, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Ensure all values are positive
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
