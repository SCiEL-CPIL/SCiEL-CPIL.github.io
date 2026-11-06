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
      3. Score each feasible node using:
         - immediate distance from current_node
         - a savings penalty: how much extra distance is added vs going straight to depot
         - a lookahead/clustering term: min distance from that node to any other feasible node or depot
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

    # Lookahead: for each feasible node, min distance to any other feasible node or depot
    # Build sub-matrix among feasible nodes + depot
    indices = np.concatenate([feasible_arr, [depot]])
    sub_dist = distance_matrix[np.ix_(indices, indices)]
    
    # For each feasible node, find min distance to any OTHER node in sub_dist (excluding self)
    lookahead = np.zeros(n_feasible)
    for i in range(n_feasible):
        # Exclude self (index i), keep all others including depot at index n_feasible
        min_dist = np.min(np.delete(sub_dist[i], i))
        lookahead[i] = min_dist

    # Score: weight immediate distance, savings penalty, and lookahead
    # Lower score = better
    alpha_savings = 0.3
    alpha_lookahead = 0.2
    scores = dist_from_current + alpha_savings * savings_penalty + alpha_lookahead * lookahead

    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
