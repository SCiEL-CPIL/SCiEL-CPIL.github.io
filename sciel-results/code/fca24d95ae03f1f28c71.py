"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 115.0
_DEADLINE = _START_TIME + _TIMEOUT

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if time.time() > _DEADLINE:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        # Fallback: Nearest feasible
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # Filter feasible nodes
    # Convert to numpy array for speed
    unvisited_arr = np.array(unvisited_nodes)
    if len(unvisited_arr) == 0:
        return 0
        
    demands_arr = np.array(demands, dtype=float)
    feasible_mask = demands_arr[unvisited_arr] <= rest_capacity
    
    if not np.any(feasible_mask):
        return 0
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # Distances
    dists_current = distance_matrix[current_node, feasible_nodes]
    dists_depot = distance_matrix[feasible_nodes, depot]
    
    # If we are at the depot, we must start a route.
    # The "savings" logic doesn't apply directly to the first node in the same way,
    # but minimizing d(0, j) is the standard nearest neighbor start.
    # However, we can also consider demand to start with a "heavy" load?
    # Usually, nearest neighbor is good for the first step to establish direction.
    if current_node == depot:
        # Pick nearest neighbor
        best_idx = np.argmin(dists_current)
        return int(feasible_nodes[best_idx])
    
    # Clarke-Wright Savings Heuristic
    # Cost to go i -> j -> depot: d(i,j) + d(j,0)
    # Cost to go i -> depot -> j: d(i,0) + d(0,j)
    # Saving = d(i,0) + d(0,j) - (d(i,j) + d(j,0))
    # We want to maximize Saving, so minimize -Saving = d(i,j) + d(j,0) - d(i,0) - d(0,j)
    # Since d(i,0) is constant for this step, we minimize: d(i,j) - d(0,j)
    
    savings_cost = dists_current - dists_depot
    
    # Check if continuing the route is beneficial
    # If the minimum cost to extend is higher than the cost to return to depot now,
    # then we should return to depot.
    # Cost to return to depot now: d(i,0)
    # Cost to extend to best j: d(i,j_best)
    # But we also have to pay d(j_best, 0) eventually.
    # The comparison is: Is d(i,j) + d(j,0) < d(i,0) + d(0,j)?
    # This is equivalent to: d(i,j) - d(0,j) < d(i,0)
    
    min_savings_cost = np.min(savings_cost)
    dist_i_depot = distance_matrix[current_node, depot]
    
    if min_savings_cost >= dist_i_depot:
        return 0
    
    # If we are extending, which node?
    # We want to minimize savings_cost.
    # Tie-breaker: Prefer higher demand to fill capacity.
    # Normalize demand to 0-1 range for scaling
    feas_demands = demands_arr[feasible_nodes]
    max_demand = feas_demands.max() if len(feas_demands) > 0 else 1.0
    
    # Add a small penalty for low demand (i.e., subtract a bonus for high demand)
    # This encourages picking larger nodes to reduce trip count.
    # The scale of distances is typically 0-100 (normalized coordinates).
    # Demand is 1-9.
    # A small coefficient like 0.1 or 0.2 should be safe.
    demand_bonus = 0.2 * (feas_demands / max_demand)
    
    final_scores = savings_cost - demand_bonus
    
    best_idx = np.argmin(final_scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
