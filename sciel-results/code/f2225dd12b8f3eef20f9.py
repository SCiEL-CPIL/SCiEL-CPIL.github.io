"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 150s limit, stop at 145s
    if time.time() - _start_time > 145:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited_arr)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Pre-compute key distances
    d_current_to_feasible = distance_matrix[current_node, feasible_nodes]
    d_feasible_to_depot = distance_matrix[feasible_nodes, depot]
    d_current_to_depot = distance_matrix[current_node, depot]
    feasible_demands = demands[feasible_nodes]
    
    # Clarke-Wright style savings:
    # savings[i] = d[current,depot] + d[i,depot] - d[current,i]
    # We want to maximize savings (minimize effective cost)
    # effective_cost = d[current,i] - beta * savings[i]
    #               = d[current,i] - beta*(d[current,depot] + d[i,depot] - d[current,i])
    #               = d[current,i]*(1+beta) - beta*(d[current,depot] + d[i,depot])
    beta = 0.5
    effective_cost = d_current_to_feasible * (1.0 + beta) - beta * (d_current_to_depot + d_feasible_to_depot)
    
    # Capacity-aware depot penalty (from inspiration 1):
    # As rest_capacity decreases, penalize nodes far from depot more heavily
    # to avoid being stranded with no feasible next node
    remaining_after = rest_capacity - feasible_demands
    min_demand = np.min(demands)
    
    # Strong penalty if after visiting this node, we can't visit any other
    stranded_mask = remaining_after < min_demand
    if np.any(stranded_mask):
        effective_cost[stranded_mask] += 500.0
    
    # Moderate penalty scaling with capacity pressure
    # penalty_weight increases as rest_capacity shrinks
    # Normalize: when rest_capacity is full (50), penalty is small; when low, large
    capacity_pressure = 1.0 / max(rest_capacity, 1.0)
    depot_penalty_weight = 3.0 * capacity_pressure
    effective_cost += depot_penalty_weight * d_feasible_to_depot
    
    # Find the minimum cost and candidates near it
    min_cost = np.min(effective_cost)
    threshold = min_cost + 0.5  # small tolerance for tie-breaking
    
    # Candidates within tolerance of best
    near_best_mask = effective_cost <= threshold
    near_best_indices = np.where(near_best_mask)[0]
    
    if len(near_best_indices) <= 1:
        best_idx = int(np.argmin(effective_cost))
        return int(feasible_nodes[best_idx])
    
    # 2-step lookahead to break ties among near-best candidates (from inspiration 3)
    # For each near-best candidate i, estimate: d[current,i] + best_followup_cost
    # where best_followup_cost is the min over j (feasible after i) of d[i,j] + d[j,depot]
    # or d[i,depot] if no feasible followup
    
    near_best_nodes = feasible_nodes[near_best_indices]
    lookahead_scores = np.zeros(len(near_best_indices))
    
    for k, idx_i in enumerate(near_best_indices):
        i = feasible_nodes[idx_i]
        d_i = d_current_to_feasible[idx_i]
        d_i_depot = d_feasible_to_depot[idx_i]
        rem_cap_after_i = rest_capacity - feasible_demands[idx_i]
        
        if rem_cap_after_i < min_demand:
            # Can't visit any more after i, so followup is just going to depot
            best_2nd = d_i_depot
        else:
            # Find best followup node j from i among all unvisited (not just feasible_nodes)
            # j must be unvisited, j != i, and demands[j] <= rem_cap_after_i
            dist_from_i = distance_matrix[i, unvisited_arr]
            unvisited_dem = demands[unvisited_arr]
            
            mask_j = (unvisited_arr != i) & (unvisited_dem <= rem_cap_after_i)
            if np.any(mask_j):
                # Cost of going i -> j -> depot
                followup_costs = dist_from_i[mask_j] + distance_matrix[unvisited_arr[mask_j], depot]
                best_2nd = np.min(followup_costs)
            else:
                best_2nd = d_i_depot
        
        lookahead_scores[k] = d_i + best_2nd
    
    # Pick the best among near-best candidates using lookahead
    best_k = int(np.argmin(lookahead_scores))
    best_idx = near_best_indices[best_k]
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
