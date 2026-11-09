"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    
    # Helper to calculate tour length
    def get_tour_length(t):
        return np.sum([distance_matrix[t[i], t[(i + 1) % n]] for i in range(n)])
    
    # Nearest Neighbor heuristic
    def nearest_neighbor(start=0):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        current = start
        
        for _ in range(n - 1):
            dists = distance_matrix[current, :].copy()
            dists[visited] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        
        return tour
    
    # 2-Opt improvement
    def two_opt(t):
        t = list(t)
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = t[i], t[i + 1]
                    c, d = t[j], t[(j + 1) % n]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
                    if delta < 0:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
            if time.time() - start_time > deadline:
                break
        return t
    
    # Or-opt improvement
    def or_opt(t):
        t = list(t)
        improved = True
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(n):
                    seg_nodes = [t[(i + k) % n] for k in range(seg_len)]
                    prev_node = t[(i - 1) % n]
                    next_node = t[(i + seg_len) % n]
                    first_seg_node = seg_nodes[0]
                    last_seg_node = seg_nodes[-1]
                    
                    remove_cost = (distance_matrix[prev_node, first_seg_node] + distance_matrix[last_seg_node, next_node]) - distance_matrix[prev_node, next_node]
                    
                    for j in range(n):
                        if i < j < i + seg_len:
                            continue
                        if j == i or j == (i + seg_len) % n:
                            continue
                        
                        p = t[(j - 1) % n]
                        q_node = t[j % n]
                        
                        insert_cost = (distance_matrix[p, first_seg_node] + distance_matrix[last_seg_node, q_node]) - distance_matrix[p, q_node]
                        
                        total_delta = remove_cost + insert_cost
                        
                        if total_delta < 0:
                            # Perform the move
                            new_t = []
                            # Remove segment
                            for k in range(n):
                                if k not in [(i + m) % n for m in range(seg_len)]:
                                    new_t.append(t[k])
                            
                            # Find insertion position
                            # j is in original tour, need to map to new_t
                            if j < i:
                                insert_pos = j
                            else:
                                insert_pos = j - seg_len
                            
                            new_t = new_t[:insert_pos] + seg_nodes + new_t[insert_pos:]
                            
                            if get_tour_length(new_t) < get_tour_length(t):
                                t = new_t
                                improved = True
                                if time.time() - start_time > deadline:
                                    return t
        return t
    
    # Find multiple good tours using different starting points
    best_tours = []
    for start in range(min(n, 10)):
        tour = nearest_neighbor(start)
        tour = two_opt(tour)
        if time.time() - start_time > deadline:
            break
        tour = or_opt(tour)
        if time.time() - start_time > deadline:
            break
        best_tours.append(tour)
    
    # Also try some random starts
    np.random.seed(42)
    for _ in range(5):
        start = np.random.randint(n)
        tour = nearest_neighbor(start)
        tour = two_opt(tour)
        if time.time() - start_time > deadline:
            break
        tour = or_opt(tour)
        if time.time() - start_time > deadline:
            break
        best_tours.append(tour)
    
    # Build heuristic matrix
    # Base: inverse distance with stronger power for sharper gradient
    diag_mask = np.eye(n, dtype=bool)
    safe_dist = distance_matrix.copy()
    safe_dist[diag_mask] = 1.0
    
    # Use a stronger power to make short edges much more attractive
    base = 1.0 / (safe_dist ** 2.0)
    
    # Boost edges that appear in good tours
    # Count how often each edge appears in the best tours
    edge_count = np.zeros((n, n))
    for tour in best_tours:
        for i in range(n):
            j = (i + 1) % n
            edge_count[tour[i], tour[j]] += 1
    
    # Boost edges that appear frequently in good tours with stronger factor
    # Normalize edge_count
    if edge_count.max() > 0:
        edge_boost = edge_count / edge_count.max()
        base *= (1.0 + 20.0 * edge_boost)
    
    # Add geometric structure boost using the distance matrix
    # Short edges in the distance matrix are likely good
    max_dist = safe_dist[~diag_mask].max()
    if max_dist > 0:
        # Use a smooth exponential falloff for short edges
        # Edges shorter than 1/4 of max get extra boost
        short_threshold = max_dist / 4.0
        short_edge_mask = (safe_dist < short_threshold) & (~diag_mask)
        # Apply exponential boost based on how short the edge is
        if np.any(short_edge_mask):
            # Normalize by threshold for smooth scaling
            normalized_short = safe_dist[short_edge_mask] / short_threshold
            boost_factor = np.exp(-3.0 * normalized_short)
            base[short_edge_mask] *= (1.0 + 5.0 * boost_factor)
    
    # Ensure diagonal is not attractive
    base[diag_mask] = 1e-9
    
    # Ensure all values are positive
    base = np.maximum(base, 1e-9)
    
    return base
# EVOLVE-BLOCK-END
