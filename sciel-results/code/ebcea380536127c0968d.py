"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 2.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated vehicle routing:
    Pick the next customer to visit, or return to depot (0) to close the route.
    
    Strategy:
    1. Filter unvisited nodes that fit within rest_capacity.
    2. If none fit, return depot.
    3. Score each feasible candidate using a combination of:
       - Immediate distance from current_node to candidate
       - 1-step lookahead: min distance from candidate to other feasible nodes or depot
       - Capacity-based penalty to encourage packing similar demands
    4. If time is running out, fall back to nearest feasible node.
    """
    if _time_exceeded():
        # Fallback: nearest feasible node
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        if len(feasible) == 1:
            return int(feasible[0])
        dists = distance_matrix[current_node, feasible]
        return int(feasible[int(np.argmin(dists))])

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
    
    # 1-step lookahead: for each candidate, min distance to any other feasible node or depot
    # This avoids picking nodes that lead to long detours or dead-ends
    dist_feas = distance_matrix[np.ix_(feasible_arr, feasible_arr)]
    np.fill_diagonal(dist_feas, np.inf)
    min_to_other_feas = np.min(dist_feas, axis=1)
    
    # Distance from each feasible node to depot
    dist_to_depot = distance_matrix[feasible_arr, depot]
    
    # Lookahead: min of going to another feasible or going to depot
    lookahead = np.minimum(min_to_other_feas, dist_to_depot)
    
    # Composite score: immediate distance + alpha * lookahead + beta * capacity_penalty
    # alpha controls the weight of lookahead; smaller alpha favors immediate proximity
    # beta controls the weight of capacity penalty; encourages packing
    alpha = 0.3
    beta = 0.1
    
    # Capacity penalty: penalize nodes that consume a large fraction of remaining capacity
    # This encourages grouping similar-sized demands together
    cap_penalty = beta * (demands[feasible_arr] / max(rest_capacity, 1))
    
    scores = immediate + alpha * lookahead + cap_penalty
    
    # Pick the candidate with the lowest score
    best_idx = int(np.argmin(scores))
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
