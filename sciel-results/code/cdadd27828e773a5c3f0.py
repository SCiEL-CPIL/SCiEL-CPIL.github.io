"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit

def _check_timeout():
    return (time.time() - _START_TIME) > _TIMEOUT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    
    Score(u) = d(current, u) + min_{v in unvisited\{u}} d(u, v)
    
    This estimates the total cost of the next two edges in the tour.
    By minimizing this, we avoid choices that lead to a long edge in the immediate future.
    """
    if len(unvisited_nodes) == 0:
        return -1

    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    if _check_timeout():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    # Vectorized implementation
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distance matrix between unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Compute min distance from each u to any other v in unvisited (excluding itself)
    # Set diagonal to inf so it doesn't affect min
    d_unvisited_for_min = d_unvisited.copy()
    np.fill_diagonal(d_unvisited_for_min, np.inf)
    
    min_dist_to_other = np.min(d_unvisited_for_min, axis=1)
    
    # Heuristic: Immediate cost + Estimated next edge cost
    score = d_current + min_dist_to_other
    
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
