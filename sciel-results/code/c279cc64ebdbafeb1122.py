"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s limit


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT


def _two_opt_path(path, start_node, end_node, dist_matrix):
    """
    Improves a path using 2-opt.
    path: list of nodes in order.
    start_node: fixed start.
    end_node: fixed end.
    Returns improved path and length.
    """
    # path is mutable list
    n = len(path)
    if n < 2:
        return path, dist_matrix[start_node, path[0]] + dist_matrix[path[-1], end_node] if n > 0 else dist_matrix[start_node, end_node]
    
    # Calculate initial length
    curr_len = dist_matrix[start_node, path[0]]
    for i in range(n - 1):
        curr_len += dist_matrix[path[i], path[i+1]]
    curr_len += dist_matrix[path[-1], end_node]
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Identify the four edges involved in reversing path[i:j+1]
                # Case 1: i > 0 and j < n-1 (internal reversal)
                # Edges: (path[i-1], path[i]) and (path[j], path[j+1])
                # New:   (path[i-1], path[j]) and (path[i], path[j+1])
                if i > 0 and j < n - 1:
                    old_cost = dist_matrix[path[i-1], path[i]] + dist_matrix[path[j], path[j+1]]
                    new_cost = dist_matrix[path[i-1], path[j]] + dist_matrix[path[i], path[j+1]]
                    if new_cost < old_cost - 1e-9:
                        path[i:j+1] = path[i:j+1][::-1]
                        curr_len += new_cost - old_cost
                        improved = True
                        # Restart or continue? Restarting is safer for correctness but slower.
                        # For small n, restarting is fine.
                        break
                # Case 2: i == 0 (involves start_node)
                elif i == 0:
                    # Edge (start, path[0]) and (path[j], path[j+1]) if j < n-1
                    # If j == n-1, edge is (path[j], end)
                    if j < n - 1:
                        old_cost = dist_matrix[start_node, path[0]] + dist_matrix[path[j], path[j+1]]
                        new_cost = dist_matrix[start_node, path[j]] + dist_matrix[path[0], path[j+1]]
                        if new_cost < old_cost - 1e-9:
                            path[0:j+1] = path[0:j+1][::-1]
                            curr_len += new_cost - old_cost
                            improved = True
                            break
                    else: # j == n-1
                        # Reverse whole path?
                        # Old: (start, path[0]) ... (path[n-2], path[n-1]) ... (path[n-1], end)
                        # New: (start, path[n-1]) ... (path[1], path[0]) ... (path[0], end)
                        old_cost = dist_matrix[start_node, path[0]] + dist_matrix[path[n-1], end_node]
                        new_cost = dist_matrix[start_node, path[n-1]] + dist_matrix[path[0], end_node]
                        if new_cost < old_cost - 1e-9:
                            path[:] = path[::-1]
                            curr_len += new_cost - old_cost
                            improved = True
                            break
                # Case 3: j == n-1 (involves end_node) and i > 0
                elif j == n - 1:
                    # Edge (path[i-1], path[i]) and (path[n-1], end)
                    old_cost = dist_matrix[path[i-1], path[i]] + dist_matrix[path[n-1], end_node]
                    new_cost = dist_matrix[path[i-1], path[n-1]] + dist_matrix[path[i], end_node]
                    if new_cost < old_cost - 1e-9:
                        path[i:] = path[i:][::-1]
                        curr_len += new_cost - old_cost
                        improved = True
                        break
        if improved:
            continue
    return path, curr_len


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return int(current_node)

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if n_unvisited == 2:
        n1, n2 = unvisited_nodes[0], unvisited_nodes[1]
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        if cost1 <= cost2:
            return int(n1)
        else:
            return int(n2)

    # If time is running out, fall back to nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Greedy with 2-Opt Lookahead
    # For each candidate, estimate the rest of the tour using NN + 2-Opt
    best_candidate = -1
    best_score = float('inf')
    
    # Pre-extract distances for speed
    # unvisited_nodes is an np.array
    
    for idx, candidate in enumerate(unvisited_nodes):
        # Create a list of remaining nodes excluding candidate
        # This is O(n)
        remaining_mask = np.ones(n_unvisited, dtype=bool)
        remaining_mask[idx] = False
        remaining = unvisited_nodes[remaining_mask]
        
        # Build a greedy NN tail
        tail = [int(candidate)]
        current_tail_node = int(candidate)
        
        # NN completion
        rem_set = set(remaining.tolist())
        # To make NN fast, we can just iterate and find min
        # But for small n, simple loop is fine
        while rem_set:
            # Find closest to current_tail_node
            # Convert to list for min
            closest = min(rem_set, key=lambda x: distance_matrix[current_tail_node, x])
            tail.append(closest)
            current_tail_node = closest
            rem_set.remove(closest)
            
        # Apply 2-Opt to the tail
        # The tail is a list of ints
        # We pass it to _two_opt_path
        # Note: _two_opt_path modifies the list in place
        tail_copy = tail[:]
        _, tail_len = _two_opt_path(tail_copy, current_node, destination_node, distance_matrix)
        
        # The total length is the length returned by _two_opt_path
        # Because _two_opt_path includes the edges to start and end
        if tail_len < best_score:
            best_score = tail_len
            best_candidate = int(candidate)
            
    return best_candidate
# EVOLVE-BLOCK-END
