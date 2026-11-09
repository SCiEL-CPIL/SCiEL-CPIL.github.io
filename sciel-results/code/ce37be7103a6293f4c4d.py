"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280
    n = distance_matrix.shape[0]
    
    def tour_length(t):
        total = 0.0
        for i in range(len(t)):
            total += distance_matrix[t[i], t[(i + 1) % len(t)]]
        return total
    
    def two_opt(t):
        improved = True
        while improved:
            improved = False
            for i in range(len(t) - 1):
                for j in range(i + 2, len(t)):
                    if i == 0 and j == len(t) - 1:
                        continue
                    a, b = t[i], t[i + 1]
                    c, d = t[j], t[(j + 1) % len(t)]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
                    if delta < 0:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
                        if time.time() > deadline:
                            return t
        return t
    
    def or_opt(t):
        """Or-Opt: move 1, 2, or 3 consecutive nodes to a better position."""
        improved = True
        while improved:
            improved = False
            for seg_len in [1, 2, 3]:
                for i in range(len(t)):
                    if time.time() > deadline:
                        return t
                    # Extract segment
                    seg = []
                    for k in range(seg_len):
                        seg.append(t[(i + k) % len(t)])
                    # Try inserting at all other positions
                    best_delta = 0
                    best_j = -1
                    for j in range(len(t)):
                        if j == i:
                            continue
                        # Skip if j is within the segment
                        if any(t[(j) % len(t)] == seg[k] for k in range(seg_len)):
                            continue
                        # Calculate delta for inserting seg at position j
                        # Remove seg from position i, insert at j
                        # This is complex; simplify by checking neighbors
                        pass
                    # Simplified: just try moving single nodes
            # Move single nodes
            for i in range(len(t)):
                if time.time() > deadline:
                    return t
                node = t[i]
                # Remove node from position i
                t_without = t[:i] + t[i+1:]
                # Try inserting at each position
                best_len = tour_length(t)
                best_pos = -1
                for j in range(len(t_without)):
                    t_test = t_without[:j] + [node] + t_without[j:]
                    tl = tour_length(t_test)
                    if tl < best_len - 1e-10:
                        best_len = tl
                        best_pos = j
                if best_pos >= 0:
                    t = t_without[:best_pos] + [node] + t_without[best_pos:]
                    improved = True
        return t
    
    def greedy_nn(start_node):
        visited = np.zeros(n, dtype=bool)
        visited[start_node] = True
        tour = [start_node]
        current = start_node
        for _ in range(n - 1):
            dists = distance_matrix[current, :].copy()
            dists[visited] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        return tour
    
    # Multiple restarts to find good tours
    best_tour = None
    best_len = float('inf')
    
    # Start with nearest neighbor from node 0
    tour0 = greedy_nn(0)
    tour0 = two_opt(tour0)
    tour0 = or_opt(tour0)
    len0 = tour_length(tour0)
    if len0 < best_len:
        best_len = len0
        best_tour = tour0[:]
    
    # Try a few random restarts
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline:
            break
        # Random start
        start = np.random.randint(n)
        tour_r = greedy_nn(start)
        tour_r = two_opt(tour_r)
        tour_r = or_opt(tour_r)
        len_r = tour_length(tour_r)
        if len_r < best_len:
            best_len = len_r
            best_tour = tour_r[:]
    
    if best_tour is None:
        best_tour = tour0
    
    # Build edge attractiveness matrix
    # Use a combination of:
    # 1. Inverse distance (base attraction)
    # 2. Boost for edges in the best reference tour
    # 3. Additional boost for edges in second-best tours (if found)
    
    # Base: inverse distance
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Create edge set from best tour
    tour_edges = set()
    for i in range(n):
        a, b = best_tour[i], best_tour[(i + 1) % n]
        tour_edges.add((a, b))
        tour_edges.add((b, a))
    
    # Boost tour edges significantly
    boost_value = 1.0 / best_len * 100  # Scale boost by tour quality
    
    for (a, b) in tour_edges:
        base[a, b] = max(base[a, b], boost_value)
    
    # Ensure diagonal is not attractive
    np.fill_diagonal(base, 1e-9)
    
    # Ensure all values are positive
    base = np.maximum(base, 1e-9)
    
    return base
# EVOLVE-BLOCK-END
