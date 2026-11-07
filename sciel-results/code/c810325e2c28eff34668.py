# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix):
    """
    Computes a heuristic matrix for ACO TSP.
    Uses a blend of inverse distance and a high-quality tour derived from NN+2-Opt restarts.
    """
    n = distance_matrix.shape[0]
    
    # 1. Base heuristic: Inverse distance
    # Add small epsilon to avoid division by zero, though diagonal is 0.
    # We set diagonal to 0 in the final heuristic because ants don't visit visited nodes,
    # and self-loops are not part of the tour.
    d = distance_matrix.copy()
    # Ensure no division by zero
    d_safe = d + 1e-6
    inv_dist = 1.0 / d_safe
    np.fill_diagonal(inv_dist, 0.0)
    
    # 2. Find a good tour using Nearest Neighbor + 2-Opt with restarts
    def _nn_tour(dist, start=0):
        n_local = dist.shape[0]
        visited = [False] * n_local
        visited[start] = True
        tour = [start]
        current = start
        for _ in range(n_local - 1):
            min_dist = float('inf')
            next_node = -1
            for j in range(n_local):
                if not visited[j] and dist[current, j] < min_dist:
                    min_dist = dist[current, j]
                    next_node = j
            if next_node == -1:
                break
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        return tour

    def _two_opt(tour, dist):
        n_local = len(tour)
        improved = True
        while improved:
            improved = False
            for i in range(1, n_local - 1):
                for j in range(i + 1, n_local):
                    if i + 1 == j:
                        continue
                    a, b = tour[i - 1], tour[i]
                    c, d_edge = tour[j - 1], tour[j]
                    old_cost = dist[a, b] + dist[c, d_edge]
                    new_cost = dist[a, c] + dist[b, d_edge]
                    if new_cost < old_cost - 1e-10:
                        tour[i:j] = tour[i:j][::-1]
                        improved = True
        return tour

    best_tour = None
    best_len = float('inf')
    
    # Try starting from a few different nodes to find a better base tour
    # 5 restarts is cheap for N=50
    start_nodes = [0, 1, 2, 3, 4]
    for s in start_nodes:
        t = _nn_tour(distance_matrix, s)
        t = _two_opt(t, distance_matrix)
        length = 0.0
        for i in range(n):
            length += distance_matrix[t[i], t[(i + 1) % n]]
        if length < best_len:
            best_len = length
            best_tour = t
            
    # 3. Create mask for edges in best_tour
    tour_mask = np.zeros((n, n), dtype=float)
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        tour_mask[u, v] = 1.0
        tour_mask[v, u] = 1.0

    # 4. Combine heuristics
    # Boost the edges of the good tour significantly
    # inv_dist provides gradient for refinement
    # tour_mask provides strong bias towards known good structure
    boost = 50.0
    heuristic = inv_dist + boost * tour_mask
    
    return heuristic
# EVOLVE-BLOCK-END
