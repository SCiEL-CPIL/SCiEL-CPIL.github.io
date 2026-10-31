# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_LIMIT = 148.0  # Safety margin before 150s kill


def _fast_mst_weight(nodes, distance_matrix):
    """Compute MST weight for a set of nodes using Prim's algorithm with numpy vectorization."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Build local distance submatrix
    local_dist = distance_matrix[np.ix_(nodes, nodes)]
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total_weight = 0.0
    
    for _ in range(n):
        # Find node with minimum distance not in MST
        masked = min_dist.copy()
        masked[in_mst] = np.inf
        best_idx = np.argmin(masked)
        
        if min_dist[best_idx] == np.inf:
            break
            
        in_mst[best_idx] = True
        total_weight += min_dist[best_idx]
        
        # Update distances using vectorized operation
        new_dists = local_dist[best_idx]
        update_mask = ~in_mst
        min_dist[update_mask] = np.minimum(min_dist[update_mask], new_dists[update_mask])
        
    return total_weight


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    TSP constructive heuristic using MST-residual-guided selection with minimax refinement.
    
    Strategy: For each candidate next node, estimate the total remaining tour cost
    as: dist(current, candidate) + alpha * MST(remaining) + beta * minimax(candidate, remaining)
    where MST captures global connectivity and minimax penalizes candidates that
    leave a very spread-out set of remaining nodes.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _LIMIT - 0.5:
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return c1 if cost1 <= cost2 else c2
    
    # Compute immediate distances
    dist_to_candidates = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    # Pre-compute distance submatrix among unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Compute minimax term: max distance from each node to any other unvisited node
    d_for_max = d_unvisited.copy()
    np.fill_diagonal(d_for_max, 0)
    max_dist_to_other = np.max(d_for_max, axis=1)
    
    # For larger sets, use MST-based residual estimation with minimax refinement
    if n_unvisited <= 40 and elapsed < _LIMIT - 2.0:
        # Weights for hybrid score
        alpha = 1.0   # Weight for MST term
        beta = 0.05   # Weight for minimax term (small penalty for spread-out remaining)
        
        scores = np.zeros(n_unvisited)
        
        for i in range(n_unvisited):
            if time.time() - _start_time > _LIMIT - 1.0:
                # Fallback: use simple scoring for remaining
                break
            
            c = unvisited_nodes[i]
            remaining = np.concatenate([unvisited_nodes[:i], unvisited_nodes[i+1:]])
            
            # Immediate cost
            immediate = dist_to_candidates[i]
            
            # MST weight of remaining nodes
            if len(remaining) == 0:
                mst_w = 0.0
            else:
                mst_w = _fast_mst_weight(remaining, distance_matrix)
            
            # Minimax cost: max distance from candidate to any remaining node
            if len(remaining) == 0:
                minimax_c = 0.0
            else:
                minimax_c = np.max(d_unvisited[i, np.concatenate([np.arange(i), np.arange(i+1, n_unvisited)])])
            
            # Additional consideration: distance from the "end" of the remaining tour back to destination
            # Approximate by min distance from any remaining node to destination
            if len(remaining) > 0:
                min_to_dest = np.min(dist_to_dest[np.concatenate([np.arange(i), np.arange(i+1, n_unvisited)])])
            else:
                min_to_dest = 0.0
            
            scores[i] = immediate + alpha * mst_w + beta * minimax_c + 0.1 * min_to_dest
        
        # Check if we completed all scores
        if not np.any(np.isnan(scores)) and time.time() - _start_time < _LIMIT - 0.5:
            best_idx = np.argmin(scores)
            return int(unvisited_nodes[best_idx])
    
    # Fallback: use detour + distance heuristic (fast)
    detours = dist_to_candidates + dist_to_dest - dist_current_to_dest
    
    # Normalize
    max_detour = np.max(detours)
    if max_detour > 1e-10:
        normalized_detours = detours / max_detour
    else:
        normalized_detours = detours
    
    max_dist = np.max(dist_to_candidates)
    if max_dist > 1e-10:
        normalized_local = dist_to_candidates / max_dist
    else:
        normalized_local = dist_to_candidates
    
    # Weighted combination
    alpha_fb = 0.6  # local distance weight
    beta_fb = 0.4   # detour weight
    
    scores = alpha_fb * normalized_local + beta_fb * normalized_detours
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
