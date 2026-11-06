"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 110.0  # Safety margin from 120s limit

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # If only 2 left, pick the one that leads to a better return to destination
        # Or simply pick nearest to current, as the loop will close via destination.
        # Actually, if 2 left, say A and B.
        # Path: ... -> current -> A -> B -> destination
        # Cost: d(current, A) + d(A, B) + d(B, destination)
        # vs ... -> current -> B -> A -> destination
        # Cost: d(current, B) + d(B, A) + d(A, destination)
        # d(A,B) is symmetric, so we compare d(current, A) + d(B, destination) vs d(current, B) + d(A, destination)
        n1, n2 = unvisited_nodes[0], unvisited_nodes[1]
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n1, destination_node]
        if cost1 <= cost2:
            return int(n1)
        else:
            return int(n2)

    # If time is running out, fall back to nearest neighbor to ensure completion
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Lookahead Nearest Neighbor
    # 1. Distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distances between unvisited nodes
    # dist_matrix_subset[i, j] is distance between unvisited[i] and unvisited[j]
    dist_subset = distance_matrix[unvisited_nodes][:, unvisited_nodes]
    
    # 3. For each candidate i, find the min distance to any other unvisited node j (j != i)
    # We set diagonal to infinity to ignore self
    np.fill_diagonal(dist_subset, np.inf)
    min_dist_to_next = np.min(dist_subset, axis=1)
    
    # 4. Total score = distance from current + estimated future cost
    # Note: This is a greedy 1-step lookahead.
    scores = dist_from_current + min_dist_to_next
    
    # 5. Select candidate with minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
