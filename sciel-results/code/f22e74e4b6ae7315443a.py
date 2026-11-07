import numpy as np

def heuristics(distance_matrix):
    """
    Computes an edge-attractiveness matrix for ACO TSP.
    Strategy: 
    1. Base heuristic is inverse distance.
    2. A reference tour is generated via Greedy Nearest Neighbor followed by 2-opt improvement.
    3. Edges in the reference tour are boosted in the heuristic matrix to guide ants.
    """
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    
    # Ensure valid input
    if not np.all(np.isfinite(D)) or D.shape[0] != D.shape[1]:
        return np.ones_like(D) / (np.max(D) + 1e-9)

    # 1. Base Heuristic: Inverse distance
    # Avoid division by zero
    D_safe = np.where(D == 0, 1e-9, D)
    H_base = 1.0 / D_safe
    H_base[np.diag_indices_from(H_base)] = 0.0

    # 2. Generate Reference Tour
    # Greedy Nearest Neighbor starting from node 0
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    current = 0
    
    for _ in range(n - 1):
        # Find nearest unvisited node
        distances_to_unvisited = D[current].copy()
        distances_to_unvisited[visited] = np.inf
        next_node = np.argmin(distances_to_unvisited)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # 3. Improve with 2-Opt
    def tour_length(tour):
        total = 0.0
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i + 1) % len(tour)]
            total += D[u, v]
        return total

    improved = True
    # Limit iterations to ensure speed, though for N=50 it converges fast
    max_iter = 100
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                # Nodes: tour[i-1], tour[i], ..., tour[j-1], tour[j], tour[j+1]
                u1 = tour[i - 1]
                u2 = tour[i]
                u3 = tour[j - 1]
                u4 = tour[j]
                
                # Current length contribution: D[u1,u2] + D[u3,u4]
                # New length contribution: D[u1,u3] + D[u2,u4]
                # Note: If i==1, u1 is tour[0]. If j==n-1, u4 is tour[n-1].
                # The reversal is between i and j inclusive.
                
                old_dist = D[u1, u2] + D[u3, u4]
                new_dist = D[u1, u3] + D[u2, u4]
                
                if new_dist < old_dist - 1e-12:
                    # Reverse segment i..j
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True

    # 4. Boost Heuristic for Edges in Reference Tour
    # Create a mask for edges in the tour
    H_final = H_base.copy()
    boost_factor = 3.0  # Multiply base heuristic by (1 + boost_factor) for tour edges
    
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        H_final[u, v] *= (1.0 + boost_factor)
        H_final[v, u] *= (1.0 + boost_factor)

    # Ensure non-negative and finite
    H_final = np.where(np.isfinite(H_final), H_final, 1e-9)
    H_final = np.maximum(H_final, 1e-9)
    
    return H_final
