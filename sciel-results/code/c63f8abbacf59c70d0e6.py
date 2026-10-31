"""TSP constructive heuristic: Regret-based selection with soft-min global structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using regret-based heuristic with soft-min global structure.
    
    For each candidate c, compute:
      regret(c) = second_best_connection(c) - best_connection(c)
      where connections are to other unvisited nodes + destination (with adaptive weight)
    
    Final score: d(cur, c) + alpha * regret(c) + beta * soft_min_future(c)
    
    Where:
      - regret prioritizes nodes that are "hard to fit in later"
      - soft_min_future uses a temperature-based soft min over future connections
      - alpha, beta adapt based on remaining nodes
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
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
    # As fewer nodes remain, destination becomes more important
    dest_weight = max(0.4, 1.0 - (n_unvisited / 35.0))
    
    # Regret weight: stronger when more nodes remain (early decisions matter more)
    alpha = 0.4 * (n_unvisited / 50.0) + 0.15
    
    # Soft-min temperature: lower temp = more greedy, higher = more exploratory
    # Use a temperature that decreases as we approach the end
    temp = 0.15 + 0.25 * (n_unvisited / 50.0)
    
    # Precompute
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For regret: compute best and second-best connections for each candidate
    # Connections include other unvisited nodes AND destination (weighted)
    sub_with_inf_diag = submatrix.copy()
    np.fill_diagonal(sub_with_inf_diag, np.inf)
    
    # Combine: for each row i, we have sub_with_inf_diag[i, :] (with inf at i) and dest_weight * dists_to_dest[i]
    # Create augmented matrix
    aug_matrix = np.hstack([sub_with_inf_diag, (dest_weight * dists_to_dest)[:, np.newaxis]])
    
    # Get two smallest per row for regret
    sorted_aug = np.sort(aug_matrix, axis=1)
    best_full = sorted_aug[:, 0]
    second_full = sorted_aug[:, 1]
    
    # Regret = second_best - best (how much worse is the alternative?)
    regrets = second_full - best_full
    
    # Proper soft-min via log-sum-exp for numerical stability
    # soft_min = -temp * log(sum(exp(-conn/temp)))
    # For numerical stability: soft_min = -temp * (max(-conn/temp) + log(sum(exp(-conn/temp - max(-conn/temp)))))
    neg_conn_over_temp = -aug_matrix / temp
    max_neg = np.max(neg_conn_over_temp, axis=1)
    # Handle potential inf in max_neg
    safe_max_neg = np.where(np.isfinite(max_neg), max_neg, 0.0)
    exp_terms = np.exp(neg_conn_over_temp - safe_max_neg[:, np.newaxis])
    log_sum = safe_max_neg + np.log(np.sum(exp_terms, axis=1))
    soft_min_values = -temp * log_sum
    
    # Final score: distance from current + regret penalty + soft-min bonus
    # Lower score is better
    # Regret: high regret means we should visit this node now (positive contribution to cost)
    # Soft-min: low soft-min means well-connected (negative contribution, i.e., bonus)
    beta = 0.35
    scores = dists_from_current + alpha * regrets - beta * soft_min_values
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
