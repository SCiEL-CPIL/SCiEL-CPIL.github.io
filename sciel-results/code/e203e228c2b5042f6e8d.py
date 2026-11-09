import numpy as np
import time

_START_TIME = None
_DEADLINE_MARGIN = 5.0

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + 150.0 - _DEADLINE_MARGIN

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    if time.time() > _get_deadline():
        # Fallback: Nearest Neighbor excluding destination if possible
        if len(unvisited_nodes) > 1:
            candidates = unvisited_nodes[unvisited_nodes != destination_node]
            if len(candidates) > 0:
                return int(candidates[np.argmin(distance_matrix[current_node, candidates])])
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(destination_node)
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Filter out destination if it's not the last node
    # We should only go to destination if it's the only one left.
    # If n_unvisited > 1, exclude destination from candidates.
    dest_mask = (unvisited_nodes == destination_node)
    
    if np.any(dest_mask) and n_unvisited > 1:
        # Exclude destination
        candidates = unvisited_nodes[~dest_mask]
    else:
        candidates = unvisited_nodes
        
    n_candidates = len(candidates)
    
    if n_candidates == 0:
        # Should only happen if n_unvisited was 1, handled above
        return int(unvisited_nodes[0])
        
    if n_candidates == 1:
        return int(candidates[0])
        
    # 3-Step Lookahead on candidates
    # d_c: (N,)
    d_c = distance_matrix[current_node, candidates]
    
    # d_uu: (N, N)
    d_uu = distance_matrix[np.ix_(candidates, candidates)]
    
    # M_v: min dist from v to any other candidate
    np.fill_diagonal(d_uu, np.inf)
    M_v = np.min(d_uu, axis=1)
    
    # K_v: min_w (d(v, w) + M_w)
    # Cost matrix C where C[v, w] = d(v, w) + M_w
    # Note: M_w is associated with w (column index in d_uu? No, d_uu[v, w] is dist v->w. M_w is min dist from w to others.
    # So we add M_w to column w of d_uu.
    Cost_vw = d_uu + M_v[np.newaxis, :]
    
    # K_v = min_w Cost_vw[v, w]
    K_v = np.min(Cost_vw, axis=1)
    
    # Score_u = d(c, u) + min_v (d(u, v) + K_v)
    # Here u is the candidate we are evaluating.
    # We need min_v (d(u, v) + K_v).
    # This is similar to the previous step.
    # Create matrix D where D[u, v] = d(u, v) + K_v
    Cost_uv = d_uu + K_v[np.newaxis, :]
    
    # min_v Cost_uv[u, v]
    min_next_cost = np.min(Cost_uv, axis=1)
    
    scores = d_c + min_next_cost
    
    best_idx = int(np.argmin(scores))
    return int(candidates[best_idx])
