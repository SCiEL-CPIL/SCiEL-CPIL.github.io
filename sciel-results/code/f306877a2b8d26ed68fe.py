import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0 # Safety margin

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) == 0:
        return -1
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if _is_deadline_passed():
        # Fallback to pure nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    # 2-step lookahead
    # 1. Distances from current to all candidates
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distances between candidates
    # D[i, j] is dist(unvisited[i], unvisited[j])
    D = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. For each candidate i, find min dist to any other candidate j != i
    # Set diagonal to inf so min doesn't pick self
    np.fill_diagonal(D, np.inf)
    
    # Min distance from each candidate to its best neighbor in the unvisited set
    min_neighbor_dists = np.min(D, axis=1)
    
    # 4. Score: dist(current, candidate) + dist(candidate, best_next_candidate)
    scores = dists_from_current + min_neighbor_dists
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
