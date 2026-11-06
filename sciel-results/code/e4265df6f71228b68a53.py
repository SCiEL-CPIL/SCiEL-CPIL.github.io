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
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
        
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
        
    # Regret-2 selection: score = (second_best_cost - best_cost) for each node
    # Higher regret means the node is uniquely well-positioned, so pick the max regret
    
    best_regret = -float('inf')
    best_node = -1
    
    # Pre-compute distance from current to all feasible nodes
    d_current_feasible = distance_matrix[current_node, feasible_nodes]
    
    for idx, j in enumerate(feasible_nodes):
        d_ij = d_current_feasible[idx]
        
        # Calculate savings: d(i, depot) + d(depot, j) - d(i, j)
        d_depot_j = distance_matrix[depot, j]
        savings = distance_matrix[current_node, depot] + d_depot_j - d_ij
        
        # Calculate lookahead: min distance from j to next feasible node or depot
        remaining_cap = rest_capacity - demands[j]
        
        # Find nodes that can be visited after j
        cap_mask = demands[unvisited_arr] <= remaining_cap
        not_j_mask = unvisited_arr != j
        
        next_feasible_mask = cap_mask & not_j_mask
        next_feasible_nodes = unvisited_arr[next_feasible_mask]
        
        if len(next_feasible_nodes) > 0:
            d_next = np.min(distance_matrix[j, next_feasible_nodes])
        else:
            d_next = distance_matrix[j, depot]
            
        # Immediate cost to visit j: d_ij + d_next - savings
        cost_j = d_ij + d_next - 1.0 * savings
        
        # For regret-2, we need the cost if we don't pick j now
        # Approximate by the cost of the next-best alternative
        # We'll compute this in a second pass
        
        if cost_j < best_regret or best_node == -1:
            # Tentatively track best cost
            pass
    
    # Two-pass regret-2: first pass computes all costs, second computes regret
    costs = np.zeros(len(feasible_nodes))
    for idx, j in enumerate(feasible_nodes):
        d_ij = d_current_feasible[idx]
        d_depot_j = distance_matrix[depot, j]
        savings = distance_matrix[current_node, depot] + d_depot_j - d_ij
        
        remaining_cap = rest_capacity - demands[j]
        cap_mask = demands[unvisited_arr] <= remaining_cap
        not_j_mask = unvisited_arr != j
        next_feasible_mask = cap_mask & not_j_mask
        next_feasible_nodes = unvisited_arr[next_feasible_mask]
        
        if len(next_feasible_nodes) > 0:
            d_next = np.min(distance_matrix[j, next_feasible_nodes])
        else:
            d_next = distance_matrix[j, depot]
            
        costs[idx] = d_ij + d_next - 1.0 * savings
    
    # For each node, regret = second_min_cost - min_cost (among all feasible nodes)
    # We want to pick the node with the highest regret
    # Sort costs to find min and second min
    sorted_costs = np.sort(costs)
    min_cost = sorted_costs[0]
    second_min_cost = sorted_costs[1]
    
    # Regret for each node: if it's the min, regret = second_min - min; else regret = min - cost (negative)
    # Actually, standard regret-2: for each candidate, regret = (cost if we don't pick it) - (cost if we do pick it)
    # Simplified: pick the node with the largest gap between its cost and the next best
    # If node j is the best, regret_j = second_min - min_cost
    # If node j is not the best, regret_j = min_cost - cost_j (which is negative)
    # So we just pick the node with the minimum cost? That's nearest neighbor.
    
    # Better regret-2: for each node j, compute the cost of the best alternative (not j)
    # regret_j = alt_cost_j - cost_j, where alt_cost_j is min cost among all nodes except j
    # We want to maximize regret_j
    
    best_node = -1
    best_regret = -float('inf')
    
    for idx, j in enumerate(feasible_nodes):
        cost_j = costs[idx]
        
        # Find min cost among all nodes except idx
        if len(feasible_nodes) > 1:
            # Create a copy of costs with this node set to inf
            costs_masked = costs.copy()
            costs_masked[idx] = float('inf')
            alt_cost = np.min(costs_masked)
        else:
            alt_cost = cost_j
        
        regret = alt_cost - cost_j
        
        if regret > best_regret:
            best_regret = regret
            best_node = j
            
    return int(best_node)
# EVOLVE-BLOCK-END
