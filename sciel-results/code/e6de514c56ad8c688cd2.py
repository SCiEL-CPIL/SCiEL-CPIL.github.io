"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP: pick next customer or depot (0) to close route.
    
    Strategy:
      1. Filter unvisited nodes that fit in rest_capacity.
      2. If none fit, return depot.
      3. Score each feasible node using a combined metric:
         - immediate distance from current_node
         - savings penalty (extra cost vs going straight to depot)
         - lookahead: min distance to any other feasible node or depot
         This balances proximity, route efficiency, and avoids dead-end picks.
    """
    if _time_exceeded():
        return 0

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return 0

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    feasible_arr = np.array(feasible, dtype=np.int64)
    n_feasible = len(feasible_arr)
    if n_feasible == 1:
        return int(feasible_arr[0])

    # Immediate distances from current_node
    dist_from_current = distance_matrix[current_node, feasible_arr]

    # Savings penalty: dist(current->node) + dist(node->depot) - dist(current->depot)
    dist_to_depot = distance_matrix[feasible_arr, depot]
    dist_current_to_depot = distance_matrix[current_node, depot]
    savings_penalty = dist_from_current + dist_to_depot - dist_current_to_depot

    # Lookahead: min distance from each feasible node to any OTHER feasible node or depot
    indices = np.concatenate([feasible_arr, [depot]])
    sub_dist = distance_matrix[np.ix_(indices, indices)]
    
    lookahead = np.zeros(n_feasible)
    for i in range(n_feasible):
        # min distance from feasible node i to any other node (including depot at index n_feasible)
        # exclude self
        min_dist = np.min(np.delete(sub_dist[i], i))
        lookahead[i] = min_dist

    # Score: combine immediate distance, savings penalty, and lookahead
    # Tuned weights: prioritize immediate proximity and route efficiency
    alpha_savings = 0.25
    alpha_lookahead = 0.35
    scores = dist_from_current + alpha_savings * savings_penalty + alpha_lookahead * lookahead

    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
