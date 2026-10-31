# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_LIMIT = 110.0  # Safety margin before 120s/150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead with adaptive destination bias and orphan penalty.
    
    Score for candidate c (index i in unvisited):
      d(cur, c) + min_{j != i} [ d(c, c_j) + gamma * d(c_j, dest) ] + lambda_orphan * max_dist(c, unvisited\{c})
    
    Structural Change:
    - Moves from 1-step lookahead (current->c->dest) to 2-step lookahead (current->c->c_j->dest).
    - Introduces an 'orphan penalty' based on the maximum distance from the candidate to other unvisited nodes.
      This prevents the tour from fragmenting into distant clusters.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > _LIMIT:
        # Fallback to nearest neighbor for safety
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        # Exact calculation for 2 nodes: cur->c1->c2->dest vs cur->c2->c1->dest
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2

    # Adaptive parameters
    # gamma: weight for distance to destination in the lookahead.
    # As n_unvisited decreases, we care more about getting close to the destination.
    # Starts low (0.0) when many nodes remain, increases to 1.0 when few remain.
    # Using a smooth transition.
    if n_unvisited > 10:
        gamma = 0.0
    else:
        # Linear ramp from 0 to 1 as n goes 10 -> 2
        gamma = (10 - n_unvisited) / 8.0 
        if gamma < 0: gamma = 0
        if gamma > 1: gamma = 1

    # lambda_orphan: weight for the orphan penalty.
    # Stronger when many nodes remain to prevent fragmentation.
    # Weakens as we get closer to the end where destination dominates.
    lambda_orphan = 0.05 * (n_unvisited / 50.0)
    
    # Precompute distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Submatrix of distances among unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # --- Orphan Penalty Calculation ---
    # For each candidate i, max distance to any other unvisited node j != i
    # Set diagonal to 0 so it doesn't affect max
    sub_for_max = submatrix.copy()
    np.fill_diagonal(sub_for_max, 0.0)
    orphan_penalties = np.max(sub_for_max, axis=1)
    
    # --- 2-Step Lookahead Calculation ---
    # We want: min_{j != i} ( d(i, j) + gamma * d(j, dest) )
    # Let M[i, j] = d(i, j) + gamma * d(j, dest)
    # Note: d(j, dest) is independent of i, so we can add it to columns of submatrix.
    
    # M is n_unvisited x n_unvisited
    # M[i, j] = submatrix[i, j] + gamma * dists_to_dest[j]
    M = submatrix + gamma * dists_to_dest[np.newaxis, :]
    
    # We need min over j != i for each i.
    # Set diagonal to inf so min skips it.
    np.fill_diagonal(M, np.inf)
    
    # inner_mins[i] = min_{j != i} M[i, j]
    inner_mins = np.min(M, axis=1)
    
    # Total score
    scores = dists_from_current + inner_mins + lambda_orphan * orphan_penalties
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
