"""Rewritten program: improved TSP constructive heuristic with 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 3 seconds before the 150s limit (safety margin)."""
    return _start_time + 145.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Use a 2-step lookahead that considers:
    1. Distance from current to candidate
    2. For each candidate, the minimum distance from that candidate to other unvisited nodes
       (to avoid isolated points)
    3. A small weight on distance to destination for end-awareness
    
    Score = d(current, cand) + beta * min_{other in unvisited\{cand}} d(cand, other) + alpha * d(cand, dest)
    """
    n = len(unvisited_nodes)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline early
    if time.time() > _get_deadline():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    # Compute distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute distances from unvisited nodes to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For 2-step lookahead: for each candidate, find the minimum distance to other unvisited nodes
    # This is O(n^2) but n <= 50, so it's fast
    # dists_between_unvisited[i,j] = d(unvisited[i], unvisited[j])
    dists_between = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each candidate i, find min distance to any other unvisited node j != i
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(dists_between, np.inf)
    min_next_dist = np.min(dists_between, axis=1)
    
    # Weighted combination
    alpha = 0.03   # Small weight for destination awareness
    beta = 0.3     # Moderate weight for 2-step lookahead (avoid isolated points)
    
    scores = dists_from_current + beta * min_next_dist + alpha * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
