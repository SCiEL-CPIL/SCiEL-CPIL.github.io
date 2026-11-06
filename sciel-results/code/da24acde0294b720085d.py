"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.monotonic()
_DEADLINE = 150.0

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Safety guard: return depot if close to deadline
    if time.monotonic() - _START > _DEADLINE - 5.0:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited_arr)
    
    # Filter feasible nodes by capacity
    demands_unvisited = demands[unvisited_arr]
    feasible_mask = demands_unvisited <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 0:
        return int(depot)
        
    if n_feasible == 1:
        return int(feasible_nodes[0])
        
    # Pre-compute distances
    d_current_feasible = distance_matrix[current_node, feasible_nodes]
    d_depot_feasible = distance_matrix[depot, feasible_nodes]
    d_current_depot = distance_matrix[current_node, depot]
    
    # Dynamic penalty: as rest_capacity decreases, prefer nodes closer to depot
    # Normalize capacity usage
    cap_usage = 1.0 - (rest_capacity / 50.0) if rest_capacity < 50.0 else 0.0
    # Penalty weight increases with capacity usage
    penalty_weight = 1.0 + 2.0 * cap_usage
    
    # Compute costs for each feasible node
    costs = np.zeros(n_feasible)
    for idx, j in enumerate(feasible_nodes):
        d_ij = d_current_feasible[idx]
        d_depot_j = d_depot_feasible[idx]
        
        # Savings: how much we save by not going back to depot
        savings = d_current_depot + d_depot_j - d_ij
        
        # Lookahead: min distance from j to next feasible node or depot
        remaining_cap = rest_capacity - demands[j]
        cap_mask = demands_unvisited <= remaining_cap
        not_j_mask = unvisited_arr != j
        next_feasible_mask = cap_mask & not_j_mask
        next_feasible_nodes = unvisited_arr[next_feasible_mask]
        
        if len(next_feasible_nodes) > 0:
            d_next = np.min(distance_matrix[j, next_feasible_nodes])
        else:
            d_next = distance_matrix[j, depot]
            
        # Cost: immediate travel + lookahead - savings (savings reduces cost)
        # Apply penalty: nodes far from depot get penalized more when capacity is tight
        depot_penalty = penalty_weight * d_depot_j
        costs[idx] = d_ij + d_next - 1.0 * savings + depot_penalty
    
    # Regret-2: for each node, regret = (min cost among others) - (cost of this node)
    # Pick node with maximum regret (uniquely best positioned)
    min_cost = np.min(costs)
    second_min_cost = np.partition(costs, 1)[1]
    
    best_node = -1
    best_regret = -float('inf')
    
    for idx, j in enumerate(feasible_nodes):
        cost_j = costs[idx]
        
        # Alt cost: min cost excluding this node
        if cost_j == min_cost:
            alt_cost = second_min_cost
        else:
            alt_cost = min_cost
        
        regret = alt_cost - cost_j
        
        if regret > best_regret:
            best_regret = regret
            best_node = j
            
    return int(best_node)
# EVOLVE-BLOCK-END
