"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # 1. Compute a high-quality reference tour
    # Start with Nearest Neighbor
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    current = 0
    
    for _ in range(n - 1):
        dists = distance_matrix[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # Helper to calculate tour length
    def get_tour_length(t):
        return sum(distance_matrix[t[i], t[(i + 1) % n]] for i in range(n))
    
    # 2-Opt improvement
    def two_opt(t):
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
            if time.time() - start_time > 290:
                break
        return t

    # Or-opt improvement (moves 1, 2, or 3 nodes)
    def or_opt(t):
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
                        if j == i or j == i + seg_len:
                            continue

                        p = t[(j - 1) % n]
                        q_node = t[j]
                        insert_cost = (distance_matrix[p, first_seg_node] + distance_matrix[last_seg_node, q_node]) - distance_matrix[p, q_node]
                        total_delta = remove_cost + insert_cost
                        
                        if total_delta < 0:
                            def do_or_move(t_list, i, j, seg_len):
                                seg = []
                                for k in range(seg_len):
                                    idx = (i + k) % n
                                    seg.append(t_list[idx])
                                
                                keep_indices = [x for x in range(n) if x not in set([(i + k) % n for k in range(seg_len)])]
                                kept_nodes = [t_list[k] for k in keep_indices]
                                
                                if j < i:
                                    insert_pos = j
                                else:
                                    insert_pos = j - seg_len
                                
                                new_list = kept_nodes[:insert_pos] + seg + kept_nodes[insert_pos:]
                                return new_list

                            new_tour = do_or_move(t, i, j, seg_len)
                            if get_tour_length(new_tour) < get_tour_length(t):
                                t = new_tour
                                improved = True
                                if time.time() - start_time > 290:
                                    return t
        return t

    # Apply 2-Opt
    tour = two_opt(tour)
    
    # Apply Or-Opt
    tour = or_opt(tour)
    
    # Alternate a few times to ensure convergence
    for _ in range(3):
        if time.time() - start_time > 290:
            break
        tour = two_opt(tour)
        tour = or_opt(tour)
        
    # 2. Build Heuristic Matrix
    diag_mask = np.eye(n, dtype=bool)
    safe_dist = distance_matrix.copy()
    safe_dist[diag_mask] = 1.0
    
    base = 1.0 / safe_dist
    
    # Boost edges in the reference tour
    for i in range(n):
        j = (i + 1) % n
        base[tour[i], tour[j]] = 1000.0
        
    # Ensure diagonal is not attractive
    base[diag_mask] = 1e-9
    
    # Ensure all values are positive
    base = np.maximum(base, 1e-9)
    
    return base
# EVOLVE-BLOCK-END
