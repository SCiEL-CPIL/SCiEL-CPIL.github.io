# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _ensure_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _time_exceeded(limit=110.0):
    _ensure_start_time()
    return (time.time() - _start_time) > limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        u1 = unvisited_nodes[0]
        u2 = unvisited_nodes[1]
        # Cost of going u1 -> u2 -> dest
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        # Cost of going u2 -> u1 -> dest
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        
        if cost1 < cost2:
            return int(u1)
        else:
            return int(u2)

    # Check time budget
    if _time_exceeded():
        # Fallback to Nearest Neighbor if time is tight
        dists = distance_matrix[current_node, unvisited_nodes]
        idx = np.argmin(dists)
        return int(unvisited_nodes[idx])

    # Structural Change: "Savings" or "2-Step-to-Destination" heuristic
    # Score = d(current, v) + d(v, destination)
    # This encourages nodes that are close to the current position AND close to the final destination,
    # effectively minimizing the "detour" and keeping the tour compact towards the end.
    
    # Vectorized calculation
    # d(current, v) for all v in unvisited
    dists_cur = distance_matrix[current_node, unvisited_nodes]
    
    # d(v, destination) for all v in unvisited
    dists_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Combined score
    scores = dists_cur + dists_dest
    
    # Select the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
