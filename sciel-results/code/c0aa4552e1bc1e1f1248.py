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
        # If only 2 left, say A and B.
        # Path: ... -> current -> A -> B -> destination
        # Cost: d(current, A) + d(A, B) + d(B, destination)
        # vs ... -> current -> B -> A -> destination
        # Cost: d(current, B) + d(B, A) + d(A, destination)
        # d(A,B) is symmetric.
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

    # 2-Step Lookahead Heuristic
    # For each candidate c in unvisited_nodes:
    # Score(c) = dist(current, c) + min_{j in unvisited \ {c}} (dist(c, j) + dist(j, destination))
    
    # 1. Distance from current to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance from all unvisited nodes to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Distance matrix between all unvisited nodes
    # dist_subset[i, j] is distance between unvisited[i] and unvisited[j]
    dist_subset = distance_matrix[unvisited_nodes][:, unvisited_nodes]
    
    # 4. Compute combined cost for each pair (i, j) where i is current candidate, j is next candidate
    # We want min over j != i of (dist_subset[i, j] + dist_to_dest[j])
    # Add dist_to_dest to each column of dist_subset
    # dist_subset is N x N. dist_to_dest is N x 1.
    # combined[i, j] = dist_subset[i, j] + dist_to_dest[j]
    combined_costs = dist_subset + dist_to_dest[None, :]
    
    # 5. For each row i, we need the min over j != i.
    # Set diagonal to infinity to exclude self
    np.fill_diagonal(combined_costs, np.inf)
    
    # 6. Find the minimum cost for each candidate i
    min_next_cost = np.min(combined_costs, axis=1)
    
    # 7. Total score for each candidate
    scores = dist_from_current + min_next_cost
    
    # 8. Select candidate with minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
