"""CVRP heuristic: 2-step look-ahead insertion with capacity-awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE = None
_TIMEOUT = 115.0  # Safety margin before 120s limit


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a 2-step look-ahead insertion heuristic.
    Returns 0 (depot) to close the route.
    
    Strategy:
    - When at depot, start the route with the nearest feasible node.
    - Otherwise, for each feasible node j, estimate the true incremental cost
      of visiting j by considering the best connection from j to another
      feasible node k (2-step look-ahead).
    - Add a capacity-awareness penalty to prefer smaller demands when tight.
    - Pick the node with the lowest total cost; close the route if none is worthwhile.
    """
    global _start_time, _DEADLINE
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes: those whose demand fits in rest_capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    # If no feasible nodes, return depot
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # If at depot, pick the nearest feasible node to start the route
    if current_node == depot:
        dists = distance_matrix[current_node, feasible_nodes]
        min_idx = np.argmin(dists)
        return int(feasible_nodes[min_idx])
    
    # Compute distances from current node to feasible nodes and to depot
    d_current_j = distance_matrix[current_node, feasible_nodes]
    d_current_depot = distance_matrix[current_node, depot]
    
    # 2-step look-ahead: for each feasible node j, find the best connection
    # from j to another feasible node k (or to depot). This estimates the
    # true incremental cost of visiting j.
    if n_feasible > 1:
        # Distance matrix among feasible nodes
        dist_jk = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
        # Distance from each feasible node to depot
        d_j_depot = distance_matrix[feasible_nodes, depot]
        
        # For each j, the best next step is min(min over k != j of dist_jk[j,k], dist_j_depot[j])
        # This represents the cost of leaving j after visiting it
        np.fill_diagonal(dist_jk, np.inf)
        best_next_cost = np.minimum(np.min(dist_jk, axis=1), d_j_depot)
    else:
        # Only one feasible node: must go to depot after
        best_next_cost = np.array([distance_matrix[feasible_nodes[0], depot]])
    
    # Incremental cost of visiting j: cost to reach j + cost to leave j
    # Compare against the cost of closing the route now (going directly to depot)
    # If we close now: cost = d_current_depot
    # If we visit j: cost = d_current_j[j] + best_next_cost[j]
    # Incremental = (d_current_j[j] + best_next_cost[j]) - d_current_depot
    incremental_cost = d_current_j + best_next_cost - d_current_depot
    
    # Capacity-awareness: when remaining capacity is tight, prefer nodes
    # with smaller demand to maximize the number of customers served per route.
    # This is especially important early in a route to avoid getting stuck.
    feasible_demands = demands[feasible_nodes]
    max_demand = 9.0
    demand_ratio = feasible_demands / max_demand
    
    # Penalty scales with how tight the capacity is
    # When rest_capacity is large relative to max demand, no penalty
    # When rest_capacity is small, stronger penalty for large demands
    capacity_tightness = 1.0 - (rest_capacity / max_demand)
    capacity_tightness = np.clip(capacity_tightness, 0.0, 1.0)
    
    # Weight the capacity penalty
    capacity_weight = 0.10
    capacity_penalty = capacity_weight * capacity_tightness * demand_ratio
    
    # Total cost: incremental routing cost + capacity penalty
    total_cost = incremental_cost + capacity_penalty
    
    # If the minimum total cost is positive, it's cheaper to close the route
    if np.min(total_cost) > 0:
        return int(depot)
    
    # Pick the node with the minimum total cost
    min_idx = np.argmin(total_cost)
    
    return int(feasible_nodes[min_idx])
# EVOLVE-BLOCK-END
