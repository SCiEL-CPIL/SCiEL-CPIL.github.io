"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0


def _elapsed_ok():
    return (time.time() - _START_TIME) < (120.0 - _DEADLINE_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: composite distance-based selection with capacity awareness.
    
    Score for candidate j:
      score_j = d(current, j) + alpha * d(j, depot) + beta * (demand_j / rest_capacity)
    
    This balances:
      - Proximity to current node (avoids backtracking)
      - Proximity to depot (keeps routes compact)
      - Demand penalty (avoids filling up with high-demand nodes when capacity is low)
    """
    if not _elapsed_ok():
        return 0
    
    if len(unvisited_nodes) == 0:
        return 0
    
    # Filter by capacity feasibility
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return 0
    
    feasible = np.array(feasible)
    
    dists_current = distance_matrix[current_node, feasible]
    dists_depot = distance_matrix[depot, feasible]
    cap_ratio = demands[feasible] / max(1.0, float(rest_capacity))
    
    # Composite score
    alpha = 0.25  # weight for depot distance
    beta = 8.0    # weight for demand penalty
    
    scores = dists_current + alpha * dists_depot + beta * cap_ratio
    
    best_idx = int(np.argmin(scores))
    best_node = int(feasible[best_idx])
    
    # Decide whether to close route: if depot is significantly closer than best customer
    # and we're running low on capacity, close the route
    dist_to_depot = distance_matrix[current_node, depot]
    min_dist_customer = np.min(dists_current)
    
    if dist_to_depot < 0.7 * min_dist_customer and rest_capacity < 10:
        return 0
    
    return best_node
# EVOLVE-BLOCK-END
