# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_LIMIT = 115.0  # Safety margin before 120s/150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    TSP constructive heuristic: hybrid of nearest-neighbor with adaptive
    destination-aware 2-step lookahead and a minimax bottleneck guard.
    
    Structural change: Replace pure sum-based scoring with a composite score
    that explicitly penalizes creating long edges (bottleneck) while still
    optimizing cumulative cost. The key insight is that in Euclidean TSP,
    tours with smaller maximum edge length tend to have smaller total length.
    
    Score for candidate i (city c_i in unvisited):
      score_i = d(cur, c_i) + min_{j!=i} [d(c_i, c_j) + gamma * d(c_j, dest)]
                + alpha * max(d(cur, c_i), min_{j!=i} d(c_i, c_j))
    
    where gamma and alpha are adaptive based on remaining nodes.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > _LIMIT:
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
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2
    if n_unvisited == 3:
        # Exhaustive check for 3 remaining nodes
        c0 = int(unvisited_nodes[0])
        c1 = int(unvisited_nodes[1])
        c2 = int(unvisited_nodes[2])
        # All 6 permutations
        perms = [
            (c0, c1, c2), (c0, c2, c1),
            (c1, c0, c2), (c1, c2, c0),
            (c2, c0, c1), (c2, c1, c0)
        ]
        best_cost = float('inf')
        best_first = c0
        for p in perms:
            cost = (distance_matrix[current_node, p[0]] + 
                    distance_matrix[p[0], p[1]] + 
                    distance_matrix[p[1], p[2]] + 
                    distance_matrix[p[2], destination_node])
            if cost < best_cost:
                best_cost = cost
                best_first = p[0]
        return best_first

    # Adaptive parameters
    # gamma: weight for destination in lookahead, increases as we near the end
    if n_unvisited > 15:
        gamma = 0.0
    else:
        gamma = min(1.0, (15 - n_unvisited) / 10.0)
    
    # alpha: weight for bottleneck penalty, stronger early to prevent long edges
    alpha = 0.25 * (n_unvisited / 50.0) + 0.03

    # Precompute distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Submatrix of distances among unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # --- 2-step lookahead score ---
    # M[i, j] = d(i, j) + gamma * d(j, dest)
    M = submatrix + gamma * dists_to_dest[np.newaxis, :]
    
    # min over j != i
    M_copy = M.copy()
    np.fill_diagonal(M_copy, np.inf)
    inner_mins = np.min(M_copy, axis=1)
    
    cumulative_scores = dists_from_current + inner_mins
    
    # --- Bottleneck penalty ---
    # For each candidate i, the max edge that would be "committed"
    sub_for_min = submatrix.copy()
    np.fill_diagonal(sub_for_min, np.inf)
    min_next_edge = np.min(sub_for_min, axis=1)
    
    # Bottleneck: max of (d(cur,i), min_next_edge[i])
    bottleneck_terms = np.stack([dists_from_current, min_next_edge], axis=1)
    bottleneck_penalties = np.max(bottleneck_terms, axis=1)
    
    # --- Composite score ---
    scores = cumulative_scores + alpha * bottleneck_penalties
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
