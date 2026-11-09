"""Capacitated VRP next-node heuristic: savings-based with capacity feasibility."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_MODULE_START = time.time()
TIME_LIMIT = 110.0  # seconds; safety margin below 120s/150s limits


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP next-node selection.
    
    Strategy: For each feasible candidate (demand fits rest_capacity), compute a
    savings-based score. The key insight is that we want to maximize the Clarke-Wright
    savings: by visiting node j next, we save distance_matrix[depot, current] + 
    distance_matrix[depot, j] - distance_matrix[current, j] compared to going back
    to depot and then to j. We want to maximize this savings, which is equivalent to
    minimizing distance_matrix[current, j] - distance_matrix[depot, j].
    
    Additionally, we add a small penalty for nodes far from the depot when capacity
    is getting tight, to avoid ending up far from depot with little capacity left.
    
    If no feasible customer exists, return depot (0).
    """
    # Deadline guard
    elapsed = time.time() - _MODULE_START
    if elapsed > TIME_LIMIT:
        if len(unvisited_nodes) == 0:
            return 0
        feasible = [u for u in unvisited_nodes if demands[u] <= rest_capacity]
        if not feasible:
            return 0
        return int(min(feasible, key=lambda u: distance_matrix[current_node, u]))

    if len(unvisited_nodes) == 0:
        return 0

    # Filter by capacity feasibility
    feasible_mask = np.array([demands[u] <= rest_capacity for u in unvisited_nodes])
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_nodes[feasible_mask]
    n_feasible = len(feasible_nodes)

    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Distances from current node to all feasible nodes
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Distances from each feasible node to depot
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Composite score:
    # Primary: minimize (dist_current - dist_depot) to maximize Clarke-Wright savings
    # Secondary: when capacity is tight, add a small penalty for distance to depot
    # This encourages extending routes to distant areas before returning to depot
    
    total_demand = np.sum(demands[unvisited_nodes])
    if total_demand > 0:
        cap_ratio = rest_capacity / total_demand
    else:
        cap_ratio = 1.0
    
    # When capacity is tight (low ratio), we should be more careful about ending
    # up far from the depot. Add a small adaptive penalty.
    w_depot = 0.3 * (1.0 - cap_ratio)  # ranges from 0 to 0.3
    
    # Score: minimize dist_from_current - dist_to_depot + w_depot * dist_to_depot
    # = dist_from_current + (w_depot - 1) * dist_to_depot
    # When w_depot is small (capacity not tight), this is approximately dist_from_current - dist_to_depot
    # which maximizes savings
    
    scores = dists_from_current - dists_to_depot + w_depot * dists_to_depot
    scores = dists_from_current + (w_depot - 1.0) * dists_to_depot
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
