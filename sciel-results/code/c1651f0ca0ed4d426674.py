"""Rewritten program: TSP constructive heuristic using regret-based savings with degree-awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Regret-based savings with degree-awareness.
    
    For each candidate u, compute a composite score combining:
    1. Immediate cost: d(current, u)
    2. Regret: how much more expensive u becomes if we pick a different node next.
       regret(u) = min_{v != u} d(current, v) + d(v, u) - d(current, u)
       High regret means u is uniquely well-positioned and should be visited now.
    3. Degree penalty: nodes with fewer unvisited neighbors are "stranded" risks.
       We penalize choosing nodes that have many alternatives (low urgency)
       and reward choosing isolated nodes (high urgency).
    4. Destination proximity: prefer nodes closer to destination to help close the tour.
    
    score(u) = d(current, u) - w1 * regret(u) + w2 * degree_penalty(u) + w3 * d(u, dest)
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance matrix among unvisited nodes
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 3. Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Compute regret for each candidate u:
    #    regret(u) = min_{v != u} [d(current, v) + d(v, u)] - d(current, u)
    #    This is the extra cost incurred if we don't visit u next.
    #    For each u, we need min over v != u of d_curr[v] + d_sub[v, u]
    
    # d_curr[:, np.newaxis] broadcasts d_curr[v] for all v
    # d_sub[v, u] is the distance from v to u
    # We want, for each u: min_v (d_curr[v] + d_sub[v, u]) where v != u
    # Transpose d_sub so that d_sub_T[u, v] = d_sub[v, u]
    d_sub_T = d_sub.T  # d_sub_T[u, v] = d_sub[v, u]
    
    # For each u, compute min over v of (d_curr[v] + d_sub_T[u, v])
    # d_curr[np.newaxis, :] has shape (1, n), d_sub_T has shape (n, n)
    # combined_regret[u, v] = d_curr[v] + d_sub_T[u, v]
    combined_regret = d_curr[np.newaxis, :] + d_sub_T
    
    # Set diagonal to inf so u doesn't consider itself
    np.fill_diagonal(combined_regret, np.inf)
    
    # min over v for each u
    min_cost_if_not_u = np.min(combined_regret, axis=1)
    
    # regret(u) = min_cost_if_not_u[u] - d_curr[u]
    # This can be negative if u is actually the best choice, which is fine
    regret = min_cost_if_not_u - d_curr
    
    # 5. Compute degree-based urgency:
    #    For each node u, count how many unvisited nodes are "close" to it.
    #    Nodes with few close neighbors are more urgent to visit.
    #    Use a threshold: count neighbors within median distance.
    
    # Median distance among unvisited pairs
    if n > 2:
        # Get upper triangle distances (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        pair_dists = d_sub[triu_indices]
        median_dist = np.median(pair_dists)
    else:
        median_dist = np.median(d_sub) if n > 1 else 1.0
    
    # For each node u, count how many other unvisited nodes are within 1.5 * median_dist
    threshold = 1.5 * median_dist
    # degree_count[u] = number of v != u such that d_sub[u, v] < threshold
    degree_count = np.sum(d_sub < threshold, axis=1)
    
    # Normalize degree count to [0, 1]
    # Low degree count = high urgency (isolated node)
    # High degree count = low urgency (well-connected node)
    if n > 1:
        degree_min = np.min(degree_count)
        degree_max = np.max(degree_count)
        if degree_max > degree_min:
            degree_norm = (degree_count - degree_min) / (degree_max - degree_min)
        else:
            degree_norm = np.zeros(n)
    else:
        degree_norm = np.zeros(n)
    
    # Degree penalty: we want to PREFER low degree_norm (isolated nodes)
    # So penalty = degree_norm (higher penalty for well-connected nodes)
    degree_penalty = degree_norm
    
    # 6. Compute destination proximity score (normalized)
    if n > 1:
        dest_min = np.min(d_dest)
        dest_max = np.max(d_dest)
        if dest_max > dest_min:
            dest_norm = (d_dest - dest_min) / (dest_max - dest_min)
        else:
            dest_norm = np.zeros(n)
    else:
        dest_norm = np.zeros(n)
    
    # 7. Compute immediate cost normalized
    curr_min = np.min(d_curr)
    curr_max = np.max(d_curr)
    if curr_max > curr_min:
        curr_norm = (d_curr - curr_min) / (curr_max - curr_min)
    else:
        curr_norm = np.zeros(n)
    
    # 8. Normalize regret
    reg_min = np.min(regret)
    reg_max = np.max(regret)
    if reg_max > reg_min:
        regret_norm = (regret - reg_min) / (reg_max - reg_min)
    else:
        regret_norm = np.zeros(n)
    
    # 9. Combine scores with weights:
    #    We want to MINIMIZE the total score.
    #    - Higher immediate cost -> worse (positive weight)
    #    - Higher regret -> better (negative weight, since high regret means we should pick it)
    #    - Higher degree_norm -> worse (positive weight, prefer isolated nodes)
    #    - Higher dest_norm -> worse (positive weight, prefer nodes closer to dest)
    
    w_immediate = 1.0
    w_regret = 1.2      # Weight for regret (encourage picking high-regret nodes)
    w_degree = 0.3      # Weight for degree urgency
    w_dest = 0.2        # Weight for destination proximity
    
    scores = (w_immediate * curr_norm 
              - w_regret * regret_norm 
              + w_degree * degree_penalty 
              + w_dest * dest_norm)
    
    # 10. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
