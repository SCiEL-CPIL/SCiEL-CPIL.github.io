import numpy as np
import time

_start_time = None
_DEADLINE_SECONDS = 115.0

def _ensure_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _time_exceeded():
    global _start_time
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE_SECONDS

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if _time_exceeded():
        d_curr = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(d_curr)
        return int(unvisited_nodes[best_idx])

    U = unvisited_nodes
    n = len(U)
    
    d_curr = distance_matrix[current_node, U]
    d_dest = distance_matrix[destination_node, U]
    sub = distance_matrix[np.ix_(U, U)]
    
    if n <= 7:
        from itertools import permutations
        best_cost = float('inf')
        best_first_node = -1
        for p in permutations(U):
            cost = distance_matrix[current_node, p[0]]
            if cost >= best_cost:
                continue
            for i in range(n - 1):
                cost += distance_matrix[p[i], p[i+1]]
                if cost >= best_cost:
                    break
            else:
                cost += distance_matrix[p[-1], destination_node]
                if cost < best_cost:
                    best_cost = cost
                    best_first_node = p[0]
        return int(best_first_node)
    
    # Compute multi-step lookahead
    M = sub + d_dest[None, :]
    np.fill_diagonal(M, np.inf)
    look_2 = np.min(M, axis=1)
    
    M3 = sub + look_2[None, :]
    np.fill_diagonal(M3, np.inf)
    look_3 = np.min(M3, axis=1)
    
    # Compute local density: for each node, min distance to other unvisited nodes
    # This penalizes choosing nodes that are isolated
    min_dist_to_other = np.min(sub, axis=1)
    
    # Compute distance to destination relative to others
    # Penalize nodes far from destination if they're also far from others
    dest_penalty = d_dest * min_dist_to_other
    
    # Composite score: balance lookahead with structural quality
    # Weight: 70% lookahead, 20% immediate, 10% structural penalty
    scores = 0.7 * look_3 + 0.2 * d_curr + 0.1 * dest_penalty
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])
