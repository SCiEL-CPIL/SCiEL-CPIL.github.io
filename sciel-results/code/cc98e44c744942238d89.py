"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 115.0


def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE


def _savings_score(current_node, depot, feasible_nodes, demands, distance_matrix):
    """
    Compute savings-based scores for each feasible node.
    Savings heuristic: visiting node i saves (d(current,i) + d(i,depot) - d(current,depot))
    relative to serving i separately from depot.
    We want to maximize savings, so we minimize negative savings.
    """
    n_feasible = len(feasible_nodes)
    if n_feasible == 0:
        return np.array([])
    
    d_current_feas = distance_matrix[current_node, feasible_nodes]
    d_feas_depot = distance_matrix[feasible_nodes, depot]
    d_current_depot = distance_matrix[current_node, depot]
    
    # Savings for visiting i: we go current -> i -> depot instead of current -> depot + depot -> i -> depot
    # Actually the standard savings is: d(current, i) + d(i, depot) - d(current, depot)
    # But since we're already at current, the "extra cost" of visiting i before returning is:
    # d(current, i) + d(i, depot) - d(current, depot)
    # We want this to be small (negative savings = detour cost)
    savings = d_current_feas + d_feas_depot - d_current_depot
    
    return savings


def _soft_min(arr, temperature=2.0):
    """
    Soft minimum that approximates min but is smoother.
    Lower temperature = sharper approximation to true min.
    """
    min_val = np.min(arr)
    # Shift to avoid overflow
    shifted = arr - min_val
    exp_vals = np.exp(-shifted / temperature)
    result = min_val + temperature * np.log(np.sum(exp_vals))
    return result


def _greedy_lookahead(current_node, depot, remaining_nodes, rest_capacity, demands, distance_matrix, max_depth=12):
    """
    Simulate the rest of the route using a greedy nearest-feasible strategy
    with savings-based tie-breaking, and return the total distance of that simulated route.
    """
    if len(remaining_nodes) == 0:
        return 0.0
    
    remaining_set = list(remaining_nodes)
    total_dist = 0.0
    current = current_node
    remaining_cap = rest_capacity
    steps = 0
    
    while remaining_set and steps < max_depth:
        steps += 1
        
        # Find feasible nodes
        feasible = [node for node in remaining_set if demands[node] <= remaining_cap]
        
        if not feasible:
            break
        
        if len(feasible) == 1:
            best_node = feasible[0]
        else:
            feasible_arr = np.array(feasible)
            d_current = distance_matrix[current, feasible_arr]
            d_depot = distance_matrix[feasible_arr, depot]
            d_current_depot = distance_matrix[current, depot]
            
            # Savings-based score: prefer nodes that minimize the detour
            # detour = d(current, i) + d(i, depot) - d(current, depot)
            # Also consider 2-step: for each i, min over j!=i of d(i,j)
            feas_dist = distance_matrix[np.ix_(feasible_arr, feasible_arr)]
            np.fill_diagonal(feas_dist, np.inf)
            min_next = np.min(feas_dist, axis=1)
            
            # Soft min for smoother lookahead
            soft_next = _soft_min(feas_dist, temperature=2.0)
            
            # Combined score: primary is immediate distance, secondary is savings, tertiary is lookahead
            detour = d_current + d_depot - d_current_depot
            scores = 1.0 * d_current + 0.3 * detour + 0.25 * soft_next
            
            best_idx = np.argmin(scores)
            best_node = feasible_arr[best_idx]
        
        total_dist += distance_matrix[current, best_node]
        current = best_node
        remaining_cap -= demands[best_node]
        remaining_set.remove(best_node)
        
        if _time_exceeded():
            break
    
    total_dist += distance_matrix[current, depot]
    
    return total_dist


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes
    unvisited_demands = demands[unvisited]
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited[feasible_mask]
    
    if _time_exceeded():
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])

    # 1. Immediate distance from current
    dists_current = distance_matrix[current_node, feasible_nodes]
    
    # 2. Distance to depot (return cost proxy)
    dists_depot = distance_matrix[feasible_nodes, depot]
    d_current_depot = distance_matrix[current_node, depot]
    
    # 3. Savings: detour cost = d(current, i) + d(i, depot) - d(current, depot)
    detour_cost = dists_current + dists_depot - d_current_depot
    
    # 4. 2-step lookahead with soft min
    feas_dist = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    np.fill_diagonal(feas_dist, np.inf)
    min_next_dist = np.min(feas_dist, axis=1)
    soft_next_dist = _soft_min(feas_dist, temperature=2.0)
    
    # 5. Cluster centrality
    dists_feas_to_unvisited = distance_matrix[np.ix_(feasible_nodes, unvisited)]
    sum_dists = np.sum(dists_feas_to_unvisited, axis=1)
    n_unvisited = len(unvisited)
    if n_unvisited > 1:
        mean_dists = sum_dists / (n_unvisited - 1)
    else:
        mean_dists = np.zeros(n_feasible)
    
    # 6. Capacity utilization bonus: prefer nodes that use more of remaining capacity
    # This encourages fuller routes, reducing number of routes
    cap_bonus = -0.06 * demands[feasible_nodes] / max(rest_capacity, 1)
    
    # Base heuristic score combining multiple factors
    base_scores = (1.0 * dists_current + 
                   0.2 * dists_depot + 
                   0.3 * detour_cost + 
                   0.25 * soft_next_dist + 
                   0.1 * mean_dists +
                   cap_bonus)
    
    # If we have enough time, use full greedy lookahead for top candidates
    # to break ties or refine the choice.
    
    # Heuristic: if n_feasible is small, we can afford lookahead.
    top_k = min(3, n_feasible)
    top_indices = np.argsort(base_scores)[:top_k]
    
    lookahead_scores = np.zeros(top_k)
    for k, idx in enumerate(top_indices):
        candidate_node = feasible_nodes[idx]
        
        rem_cap = rest_capacity - demands[candidate_node]
        
        if rem_cap <= 0:
            lookahead_scores[k] = distance_matrix[current_node, candidate_node] + distance_matrix[candidate_node, depot]
        else:
            rem_nodes_list = [n for n in unvisited_nodes if n != candidate_node]
            lookahead_scores[k] = _greedy_lookahead(candidate_node, depot, rem_nodes_list, rem_cap, demands, distance_matrix, max_depth=12)
        
        if _time_exceeded():
            # Fill remaining with large value
            for j in range(k+1, top_k):
                lookahead_scores[j] = float('inf')
            break
    
    # Combine base score and lookahead score
    # Weight the lookahead heavily since it captures the full route cost
    # Use a blended score: normalize both to comparable scales
    if np.all(lookahead_scores < float('inf')):
        # Normalize base scores to [0, 1] range
        base_min = np.min(base_scores[top_indices])
        base_max = np.max(base_scores[top_indices])
        if base_max > base_min:
            base_norm = (base_scores[top_indices] - base_min) / (base_max - base_min)
        else:
            base_norm = np.zeros(top_k)
        
        # Normalize lookahead scores to [0, 1] range
        la_min = np.min(lookahead_scores)
        la_max = np.max(lookahead_scores)
        if la_max > la_min:
            la_norm = (lookahead_scores - la_min) / (la_max - la_min)
        else:
            la_norm = np.zeros(top_k)
        
        # Blend: 60% lookahead, 40% base
        combined = 0.6 * la_norm + 0.4 * base_norm
    else:
        combined = base_scores[top_indices]
    
    best_local_idx = np.argmin(combined)
    best_global_idx = top_indices[best_local_idx]
    
    return int(feasible_nodes[best_global_idx])
# EVOLVE-BLOCK-END
