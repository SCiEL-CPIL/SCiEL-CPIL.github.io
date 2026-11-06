"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated vehicle routing:
    Pick the next customer to visit, or return to depot (0) to close the route.
    
    Strategy:
    1. Filter unvisited nodes that fit within rest_capacity.
    2. If none fit, return depot.
    3. If current_node is depot, must pick a customer (unless no feasible).
    4. Score each feasible candidate using a multi-criteria heuristic.
    5. Compare best candidate score vs depot return score.
    """
    if len(unvisited_nodes) == 0:
        return int(depot)

    # Filter feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return int(depot)
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    n_feas = len(feasible_arr)
    
    # Immediate distance from current node to each feasible candidate
    immediate = distance_matrix[current_node, feasible_arr]
    
    # If we are at the depot, we cannot return to the depot. We must pick a customer.
    if current_node == depot:
        # Pick the nearest feasible node to start the route
        best_idx = int(np.argmin(immediate))
        return int(feasible_arr[best_idx])

    if _time_exceeded():
        # Fallback: nearest feasible node
        best_idx = int(np.argmin(immediate))
        return int(feasible_arr[best_idx])

    # Lookahead: soft-min distance to any other feasible unvisited node or the depot.
    # Using a log-sum-exp (soft-min) instead of a hard min smooths the penalty so
    # that candidates with several nearby options (not just the single closest) are
    # rewarded, and the objective is differentiable-friendly (no hard kinks).
    
    # Vectorized: get submatrix of distances between feasible nodes
    feas_dist = distance_matrix[np.ix_(feasible_arr, feasible_arr)]  # (n, n)
    dist_to_depot = distance_matrix[feasible_arr, depot]  # (n,)
    
    # For each candidate i, soft-min over other feasible nodes (excluding self)
    np.fill_diagonal(feas_dist, np.inf)
    # Soft-min via log-sum-exp:  -logsumexp(-x)  approximates min(x) as beta->0
    beta = 0.05  # small => close to hard min but smooth
    shifted = -beta * feas_dist
    row_max = np.max(shifted, axis=1)
    logsumexp_row = row_max + np.log(np.sum(np.exp(shifted - row_max[:, None]), axis=1))
    soft_min_to_other_feasible = -logsumexp_row / beta  # (n,)
    
    # Soft-min between "distance to other feasible" and "distance to depot"
    # Combine by taking soft-min over the two options per candidate
    two_options = np.column_stack([soft_min_to_other_feasible, dist_to_depot])  # (n, 2)
    shifted2 = -beta * two_options
    row_max2 = np.max(shifted2, axis=1)
    logsumexp2 = row_max2 + np.log(np.sum(np.exp(shifted2 - row_max2[:, None]), axis=1))
    lookahead_cost = -logsumexp2 / beta  # (n,)
    
    # Capacity utilization: prefer nodes with higher demand to fill trucks
    # This reduces the number of routes needed
    demand_ratio = demands[feasible_arr] / max(rest_capacity, 1)
    
    # Composite score:
    # - Primary: immediate distance (minimize)
    # - Secondary: lookahead cost (minimize, avoid dead-ends)
    # - Tertiary: capacity utilization (maximize, so subtract)
    
    alpha = 0.55
    gamma = 0.12
    
    scores = immediate + alpha * lookahead_cost - gamma * demand_ratio
    
    # Also consider returning to depot: compare depot "score" vs best candidate
    dist_to_depot_from_current = distance_matrix[current_node, depot]
    
    # Estimate: if we return now, how much capacity is "wasted"?
    # We want to discourage early returns when the truck is still fairly full,
    # but not so strongly that we force long detours. Use a gentler, bounded
    # penalty that scales with how much capacity is left unused relative to
    # typical remaining demand.
    if len(unvisited_nodes) > 0:
        avg_remaining_demand = float(np.mean(demands[unvisited_nodes]))
        # Fraction of capacity that is "free" relative to a typical next demand.
        # If rest_capacity is large compared to avg demand, returning now wastes
        # a lot of a vehicle's trip; penalize proportionally but cap it.
        free_ratio = rest_capacity / max(avg_remaining_demand, 1.0)
        # Penalize only the excess above 1 (i.e., more than one more customer fits).
        excess = max(0.0, free_ratio - 1.0)
        depot_score = dist_to_depot_from_current + 0.15 * min(excess, 3.0) * 2.0
    else:
        depot_score = dist_to_depot_from_current
    
    best_score = float(np.min(scores))
    
    if depot_score < best_score:
        return int(depot)
    
    # Pick the candidate with the lowest score
    best_idx = int(np.argmin(scores))
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
