"""Rewritten program: improved TSP constructive heuristic with soft-min lookahead and adaptive destination bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 2 seconds before the 120s limit (safety margin)."""
    return _start_time + 118.0

def _soft_min(values, temperature=1.0):
    """Compute soft-min of an array of values using log-sum-exp trick.
    
    soft_min = -1/t * log(sum(exp(-t * v_i)))
    As t -> inf, this approaches min(v). For t=1, it's a smooth approximation.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    scaled = -temperature * values
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    return -log_sum_exp / temperature

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Use soft-min 2-step lookahead combined with adaptive destination bias.
    
    Score for candidate c:
      d(current, c) + alpha(n) * d(c, dest) + beta * softmin_{c' != c} d(c, c')
    
    where alpha(n) increases as fewer nodes remain, and softmin provides a smooth
    surrogate for the minimum outgoing edge cost from c.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if time.time() > _get_deadline():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive alpha: stronger destination bias when fewer nodes remain
    alpha = 0.02 + 0.15 * (1.0 - n_unvisited / 50.0)
    
    scores = dists_from_current + alpha * dists_to_dest
    
    # Soft-min 2-step lookahead
    unvisited_idx = unvisited_nodes.astype(int)
    sub_matrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    diag_idx = np.arange(n_unvisited)
    sub_matrix_no_diag = sub_matrix.copy()
    sub_matrix_no_diag[diag_idx, diag_idx] = 1e10
    
    temp = 5.0
    soft_min_values = np.array([
        _soft_min(sub_matrix_no_diag[i], temperature=temp)
        for i in range(n_unvisited)
    ])
    
    beta = 0.3
    scores += beta * soft_min_values
    
    if time.time() > _get_deadline():
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
