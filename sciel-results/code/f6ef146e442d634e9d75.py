"""TSP constructive heuristic: Hybrid 2-step lookahead with MST-compactness structural penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a hybrid of 2-step lookahead and structural compactness.
    
    Structural Change:
    Instead of relying solely on path lookahead, we introduce a structural penalty based on
    the connectivity of the candidate node to the remaining unvisited cluster. This mimics
    the behavior of maintaining a compact Minimum Spanning Tree (MST) structure for the
    unvisited nodes.
    
    Score for candidate c:
      path_score(c) + lambda_struct * connectivity_penalty(c)
      
    Where:
      path_score(c) = d(cur, c) + min_{b != c} [ d(c, b) + gamma * d(b, dest) ]
      connectivity_penalty(c) = distance from c to its nearest neighbor in unvisited_nodes
      gamma = destination awareness, increases as nodes run out
      lambda_struct = weight for structural compactness, adaptive
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Handle small cases with exact or simple logic
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2

    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])

    # Adaptive parameters
    # gamma: destination awareness, increases as nodes run out
    # We want to start looking towards destination earlier to close the loop
    gamma = max(0.0, 1.0 - (n_unvisited / 25.0))
    
    # lambda_struct: weight for structural compactness (MST proxy)
    # Stronger when many nodes remain to keep the cluster tight
    # Weaker when few nodes remain because the path constraint dominates
    lambda_struct = 0.15 * (n_unvisited / 50.0)
    
    # Precompute distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Submatrix of distances among unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # --- Structural Component: Nearest Neighbor in Remaining Set ---
    # For each candidate i, find the minimum distance to any other unvisited node j != i
    # Set diagonal to inf so it doesn't pick itself
    submatrix_for_nn = submatrix.copy()
    np.fill_diagonal(submatrix_for_nn, np.inf)
    # If n_unvisited is 1, this would be inf, but we handled n<=2 above.
    # If n_unvisited is 2, min is the other node.
    nearest_neighbor_dists = np.min(submatrix_for_nn, axis=1)
    
    # --- Path Component: 2-step Lookahead ---
    # M[i, j] = d(i, j) + gamma * d(j, dest)
    M = submatrix + gamma * dists_to_dest[np.newaxis, :]
    
    # We need min over j != i for each i.
    M_copy = M.copy()
    np.fill_diagonal(M_copy, np.inf)
    
    # inner_mins[i] = min_{j != i} M[i, j]
    inner_mins = np.min(M_copy, axis=1)
    
    # Path score: d(cur, i) + inner_mins[i]
    path_scores = dists_from_current + inner_mins
    
    # --- Combine Scores ---
    # Total score = Path Score + lambda * Structural Penalty
    total_scores = path_scores + lambda_struct * nearest_neighbor_dists
    
    best_idx = np.argmin(total_scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
