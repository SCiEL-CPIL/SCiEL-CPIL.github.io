"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of TSP.
    
    Strategy: Use a Minimum Spanning Tree (MST) based heuristic combined with
    nearest-neighbor tour structure.
    1. Compute the MST of the complete graph defined by the distance matrix.
    2. Compute a nearest-neighbor tour starting from node 0.
    3. Assign high attractiveness to edges in both MST and NN tour.
    4. Use inverse distance with a tunable exponent for base attractiveness.
    """
    n = distance_matrix.shape[0]
    
    # 1. Compute MST using Prim's algorithm
    visited = np.zeros(n, dtype=bool)
    min_dist = np.inf * np.ones(n)
    parent = np.zeros(n, dtype=int)
    
    min_dist[0] = 0.0
    visited[0] = True
    parent[0] = -1
    
    mst_edges = np.zeros((n, n), dtype=bool)
    
    for _ in range(n - 1):
        temp_dist = min_dist.copy()
        temp_dist[visited] = np.inf
        u = np.argmin(temp_dist)
        
        if min_dist[u] == np.inf:
            break
            
        visited[u] = True
        
        if parent[u] != -1:
            mst_edges[u, parent[u]] = True
            mst_edges[parent[u], u] = True
            
        for v in range(n):
            if not visited[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                parent[v] = u
                
    # 2. Compute nearest-neighbor tour from node 0
    nn_tour = [0]
    unvisited = set(range(1, n))
    current = 0
    
    while unvisited:
        # Find nearest unvisited node to current
        nearest = min(unvisited, key=lambda v: distance_matrix[current, v])
        nn_tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    # Add edge back to start
    nn_tour.append(0)
    
    # Create NN tour edge mask
    nn_edges = np.zeros((n, n), dtype=bool)
    for i in range(len(nn_tour) - 1):
        nn_edges[nn_tour[i], nn_tour[i + 1]] = True
        nn_edges[nn_tour[i + 1], nn_tour[i]] = True
    
    # 3. Construct Heuristic
    # Base heuristic: Inverse distance with exponent 1.5
    # This gives moderate preference for short edges without being too extreme
    base = 1.0 / (distance_matrix + 1e-9) ** 1.5
    
    # Boost MST edges and NN tour edges
    heuristic = base.copy()
    heuristic[mst_edges] *= 8.0
    heuristic[nn_edges] *= 8.0
    
    # For edges in both MST and NN tour, apply a stronger boost
    both = mst_edges & nn_edges
    heuristic[both] *= 1.5  # Additional boost on top of the 8x already applied
    
    # Zero out diagonal
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
