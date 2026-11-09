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
    
    # Clarke-Wright Savings with angular direction bonus
    # savings_cost = d(i,j) - d(0,j); lower is better
    savings_cost = dists_current - dists_depot
    
    # Check if continuing the route is beneficial
    min_savings_cost = np.min(savings_cost)
    dist_i_depot = distance_matrix[current_node, depot]
    
    if min_savings_cost >= dist_i_depot:
        return 0
    
    # Compute angular direction from depot to each feasible node
    # and from current node to each feasible node
    # We want to continue in a similar direction to reduce backtracking
    n_feasible = len(feasible_nodes)
    
    if n_feasible > 1:
        # Get coordinates from distance matrix (assuming Euclidean)
        # Reconstruct approximate angles using the distance matrix
        # Angle at depot: direction from depot to node j
        # Angle at current: direction from current to node j
        
        # Use the distance matrix to compute relative directions
        # For each feasible node j, compute the angle between:
        # - vector depot->current and depot->j
        # - vector current->depot and current->j
        
        # Simplified: use the savings itself as a proxy for direction alignment
        # Nodes with high savings are naturally in the "same direction" as current
        # from the depot's perspective
        
        # Add a small angular clustering bonus based on how well the node
        # aligns with the current route direction
        # Use the ratio of d(current, j) to d(depot, j) as a direction indicator
        # If current is "behind" j relative to depot, this ratio is high
        
        # Compute a direction score: prefer nodes that extend the current route
        # direction rather than doubling back
        # d(current,j) / (d(current,depot) + d(depot,j)) - 1
        # This is -1 when j is opposite to depot from current, 0 when aligned
        
        denom = dist_i_depot + dists_depot
        denom = np.where(denom > 1e-10, denom, 1e-10)
        direction_score = dists_current / denom - 1.0  # range [-1, 0]
        
        # Prefer nodes with lower direction_score (more aligned with route direction)
        # Scale: direction_score ranges from -1 to 0, so multiply by a small factor
        angular_bonus = 0.3 * direction_score
        
        # Demand bonus to fill capacity
        feas_demands = demands_arr[feasible_nodes]
        max_demand = feas_demands.max() if len(feas_demands) > 0 else 1.0
        demand_bonus = 0.2 * (feas_demands / max_demand)
        
        final_scores = savings_cost + angular_bonus - demand_bonus
    else:
        final_scores = savings_cost.copy()
    
    best_idx = np.argmin(final_scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
