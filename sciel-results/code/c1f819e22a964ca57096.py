"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline tracking
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _check_deadline(safety_margin=5.0):
    """Return True if we should stop and return depot."""
    elapsed = time.time() - _get_start_time()
    return elapsed > (120.0 - safety_margin)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return 0

    # Deadline guard
    if _check_deadline():
        return 0

    dm = np.asarray(distance_matrix)
    demands_arr = np.asarray(demands)

    # 1. Filter feasible nodes: demand[node] <= rest_capacity
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands_arr[node] <= rest_capacity:
            feasible_nodes.append(node)
            
    if len(feasible_nodes) == 0:
        return 0

    feasible_np = np.array(feasible_nodes)
    n_feasible = len(feasible_np)

    # 2. Savings-based heuristic with 2-step lookahead
    # For each candidate i, estimate the cost impact:
    #   cost_if_pick_i = dm[current, i] + (estimated remaining cost after picking i)
    #   vs. cost_if_return = dm[current, depot] (close route now)
    #
    # The key insight: picking node i "saves" the depot round-trip for the
    # cluster around i. We estimate savings as:
    #   saving_i = dm[depot, i] + dm[i, nearest_neighbor] - dm[depot, nearest_neighbor]
    # where nearest_neighbor is the closest feasible node to i (excluding i).
    #
    # We combine: minimize dm[current, i] - alpha * saving_i
    # This encourages picking nodes that have high savings (well-connected clusters)
    # while also considering proximity to current position.

    # Distance from current to all feasible
    dist_current = dm[current_node, feasible_np]

    # Compute savings for each feasible node
    # For each node i, find its nearest feasible neighbor j (j != i)
    # saving_i = dm[depot, i] + dm[i, j] - dm[depot, j]
    
    # Distance matrix among feasible nodes
    dm_feasible = dm[np.ix_(feasible_np, feasible_np)]
    dist_depot_to_feasible = dm[depot, feasible_np]

    # For each node, find nearest neighbor (excluding self)
    # Set diagonal to inf to exclude self
    dm_feasible_diag_inf = dm_feasible.copy()
    np.fill_diagonal(dm_feasible_diag_inf, np.inf)
    
    # Nearest neighbor index for each feasible node
    nn_indices = np.argmin(dm_feasible_diag_inf, axis=1)
    nn_distances = dm_feasible_diag_inf[np.arange(n_feasible), nn_indices]
    
    # Savings: picking i saves the separate depot round-trip
    # saving_i = dm[depot, i] + dm[i, nn] - dm[depot, nn]
    nn_depot_dist = dist_depot_to_feasible[nn_indices]
    savings = dist_depot_to_feasible + nn_distances - nn_depot_dist

    # Weighted score: proximity minus savings benefit
    # alpha controls how much we value savings vs. proximity
    alpha = 0.5
    
    scores = dist_current - alpha * savings

    # Also consider: if rest_capacity is low, prefer returning to depot
    # to avoid getting stuck. Scale alpha down when capacity is tight.
    total_demand_remaining = np.sum(demands_arr[feasible_np])
    if total_demand_remaining > 0:
        capacity_ratio = rest_capacity / max(1.0, float(total_demand_remaining))
        # If capacity is very tight relative to remaining demand, reduce savings weight
        if capacity_ratio < 0.5:
            alpha *= capacity_ratio * 2.0

    # Re-compute scores with adjusted alpha
    scores = dist_current - alpha * savings

    # Handle case where all savings are negative (picking any node increases cost)
    # In that case, compare against returning to depot
    best_score = np.min(scores)
    return_cost = dm[current_node, depot]
    
    # If the best candidate's adjusted cost is worse than returning, consider returning
    # But we need to be careful: returning closes the route, so we only return
    # if it's clearly better OR if we have no other good option.
    # Use a threshold: return only if best candidate is significantly worse
    if best_score > return_cost * 1.2:
        # Check if it's worth returning vs. picking the least bad option
        # If capacity is very low, prefer returning
        if rest_capacity < 5:
            return 0
        # Otherwise, still pick the best candidate (we need to visit all nodes)
    
    min_idx = np.argmin(scores)
    
    return int(feasible_np[min_idx])
# EVOLVE-BLOCK-END
