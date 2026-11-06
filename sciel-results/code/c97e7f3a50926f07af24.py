"""Strategy: Batch-solve the full TSP on first call using multi-restart 2-opt + Or-opt, then replay the tour."""
import numpy as np
import time

# Global state
_solved_tour = None
_start_time = None
_TIME_LIMIT = 145.0

def _tour_length(tour, dist_matrix):
    """Compute closed tour length."""
    if len(tour) <= 1:
        return 0.0
    n = len(tour)
    return np.sum(dist_matrix[tour[:-1], tour[1:]]) + dist_matrix[tour[-1], tour[0]]

def _two_opt(tour, dist_matrix, max_passes=100):
    """Iterative 2-opt improvement."""
    n = len(tour)
    if n < 4:
        return tour
    t = np.array(tour, dtype=int)
    best_len = _tour_length(t, dist_matrix)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # same edge
                # Compute delta
                a, b = t[i], t[i+1]
                c, d = t[j], t[(j+1) % n]
                d_old = dist_matrix[a, b] + dist_matrix[c, d]
                d_new = dist_matrix[a, c] + dist_matrix[b, d]
                if d_new < d_old - 1e-8:
                    t[i+1:j+1] = t[i+1:j+1][::-1]
                    best_len = d_new - d_old + best_len
                    improved = True
        if not improved:
            break
    return t

def _or_opt(tour, dist_matrix, max_passes=50):
    """Or-opt: move 1, 2, or 3 consecutive nodes to a different position."""
    n = len(tour)
    if n < 5:
        return tour
    t = np.array(tour, dtype=int)
    best_len = _tour_length(t, dist_matrix)
    
    for _ in range(max_passes):
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(n):
                # Extract segment
                seg = t[i:i+seg_len].copy()
                if len(seg) < seg_len:
                    seg = np.concatenate([seg, t[:seg_len - len(seg)]])
                # Remove segment
                if i + seg_len <= n:
                    remaining = np.concatenate([t[:i], t[i+seg_len:]])
                else:
                    remaining = np.concatenate([t[:i], t[i+seg_len-n:]])
                
                if len(remaining) <= 1:
                    continue
                
                m = len(remaining)
                # Try inserting after each position in remaining
                for j in range(m):
                    # Insert seg after remaining[j]
                    new_tour = np.concatenate([remaining[:j+1], seg, remaining[j+1:]])
                    new_len = _tour_length(new_tour, dist_matrix)
                    if new_len < best_len - 1e-8:
                        t = new_tour
                        best_len = new_len
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return t

def _solve_tsp(dist_matrix, time_limit):
    """Multi-restart local search: NN starts + 2-opt + Or-opt."""
    n = dist_matrix.shape[0]
    deadline = time.time() + time_limit
    best_tour = None
    best_len = float('inf')
    
    # Multiple random + NN starts
    n_starts = 30
    for start_idx in range(n_starts):
        if time.time() > deadline - 2:
            break
        
        if start_idx == 0:
            # Nearest neighbor from node 0
            start = 0
        elif start_idx == 1:
            # Nearest neighbor from node n//2
            start = n // 2
        elif start_idx == 2:
            # Farthest from 0
            start = int(np.argmax(dist_matrix[0]))
        else:
            start = start_idx % n
        
        # Build NN tour
        unvisited = set(range(n))
        unvisited.remove(start)
        tour = [start]
        curr = start
        while unvisited:
            # Nearest neighbor
            dists = dist_matrix[curr, list(unvisited)]
            next_node = list(unvisited)[np.argmin(dists)]
            tour.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        
        tour_arr = np.array(tour, dtype=int)
        
        # Apply 2-opt
        tour_arr = _two_opt(tour_arr, dist_matrix, max_passes=50)
        
        # Apply Or-opt
        tour_arr = _or_opt(tour_arr, dist_matrix, max_passes=20)
        
        # Another 2-opt pass after Or-opt
        tour_arr = _two_opt(tour_arr, dist_matrix, max_passes=30)
        
        tour_len = _tour_length(tour_arr, dist_matrix)
        if tour_len < best_len:
            best_len = tour_len
            best_tour = tour_arr.copy()
    
    return best_tour, best_len

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _solved_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _solved_tour = None
    
    # If we haven't solved yet, do it now (only on first call or early calls)
    if _solved_tour is None:
        elapsed = time.time() - _start_time
        remaining_time = _TIME_LIMIT - elapsed
        if remaining_time > 5:
            _solved_tour, _ = _solve_tsp(distance_matrix, remaining_time - 2)
        else:
            # Fallback: simple NN
            _solved_tour = np.array([current_node] + list(unvisited_nodes), dtype=int)
    
    # Now replay the pre-computed tour
    # Find where current_node is in the solved tour
    # The tour is a cycle. We need to return the next node after current_node.
    
    if len(unvisited_nodes) == 0:
        return -1
    
    # Find current_node's position in the tour
    # The tour is a list of all n nodes in order
    # We need to find what comes after current_node in the tour
    
    # Get the sequence: find index of current_node
    tour_list = list(_solved_tour)
    n = len(tour_list)
    
    # Find current_node in the tour
    try:
        idx = tour_list.index(current_node)
    except ValueError:
        # Fallback: nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # The next node in the tour cycle
    next_in_tour = tour_list[(idx + 1) % n]
    
    # If that node is still unvisited, return it
    if next_in_tour in unvisited_nodes:
        return int(next_in_tour)
    else:
        # Fallback: nearest unvisited
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
