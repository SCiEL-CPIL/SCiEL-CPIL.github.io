"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_MARGIN = 10.0  # seconds before 150s limit


def _check_deadline():
    """Return True if deadline is approaching."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def _compute_connectivity_score(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    For each candidate node in unvisited_nodes, compute a connectivity-aware score
    that approximates the cost of completing a degree-2 tour through remaining nodes.
    
    The score for candidate c is:
      dist(current, c) + min_dist(c, remaining) + min_dist(c, destination)
    
    This penalizes candidates that are far from both the rest of the unvisited set
    and the destination, helping to avoid disconnected subtours.
    """
    n = len(unvisited_nodes)
    if n == 0:
        return np.array([])
    
    # dist from current to each candidate
    dist_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, compute min distance to other unvisited nodes
    # and min distance to destination
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    scores = np.zeros(n)
    for i in range(n):
        # min distance from candidate i to any other unvisited node (excluding itself)
        other_dists = sub_dist[i].copy()
        other_dists[i] = np.inf
        min_to_other = np.min(other_dists) if n > 1 else 0.0
        
        # min distance to destination
        min_to_dest = dist_to_dest[i]
        
        # Total score: immediate edge + future connection costs
        scores[i] = dist_current[i] + min_to_other + min_to_dest
    
    return scores


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using a connectivity-aware heuristic:
    Score each candidate by the sum of:
    - distance from current node
    - minimum distance to any other unvisited node
    - distance to destination
    
    This global connectivity signal helps prevent the creation of
    disconnected subtours and ensures the partial tour can be closed efficiently.
    """
    if _check_deadline():
        # Safety fallback: nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Compute connectivity scores for all candidates
    scores = _compute_connectivity_score(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Select candidate with minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
