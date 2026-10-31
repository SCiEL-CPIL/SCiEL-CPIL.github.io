"""TSP constructive heuristic: 2-step lookahead with adaptive destination bias and orphan penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using 2-step lookahead with adaptive destination bias and orphan penalty.
    
    Score for candidate c:
      d(cur, c) + min_{b != c} [ d(c, b) + gamma * d(b, dest) ] + lambda_orphan * orphan_penalty(c)
    
    Where:
      - gamma increases as few nodes remain (destination awareness)
      - orphan_penalty(c) = max_{j != c} d(c, j) over unvisited nodes (bottleneck guard)
      - lambda_orphan is small, scales with remaining nodes to prevent stranding
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
    # gamma: destination awareness, increases as nodes run out
    gamma = max(0.0, 1.0 - (n_unvisited / 20.0))
    
    # lambda_orphan: orphan penalty weight, stronger when more nodes remain
    # (early on, stranding a node is costly; late on, destination dominates)
    lambda_orphan = 0.08 * (n_unvisited / 50.0)
    
    # Precompute
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Orphan penalty: for each candidate i, max distance to any other unvisited node
    # Set diagonal to 0 so it doesn't affect max
    sub_for_max = submatrix.copy()
    np.fill_diagonal(sub_for_max, 0.0)
    orphan_penalties = np.max(sub_for_max, axis=1)
    
    # 2-step lookahead: for each candidate i, find min over b != i of [d(i,b) + gamma * d(b, dest)]
    # M[b, a] = d(b, a) + gamma * d(a, dest), with diagonal inf
    M = submatrix + gamma * dists_to_dest[np.newaxis, :]
    np.fill_diagonal(M, np.inf)
    
    # For each candidate i, we need min over b != i of M[i, b]
    # But we also need to exclude b = i, which is already handled by diagonal inf.
    # However, we also want to exclude the case where b = i in the min, which diagonal inf handles.
    # But wait: we're computing min over b for each i. M[i, b] for b != i is valid.
    # M[i, i] = inf, so min will skip it.
    
    # inner_min[i] = min_{b != i} M[i, b] = min_{b != i} (d(i,b) + gamma * d(b, dest))
    inner_mins = np.min(M, axis=1)
    
    # Total score for each candidate i:
    # score[i] = d(cur, i) + inner_min[i] + lambda_orphan * orphan_penalties[i]
    scores = dists_from_current + inner_mins + lambda_orphan * orphan_penalties
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
