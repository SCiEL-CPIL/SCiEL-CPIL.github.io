"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
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
    3. Score each feasible candidate using a lookahead heuristic:
       - Immediate distance from current_node to candidate
       - Plus a fraction of the minimum distance from candidate to any other
         feasible unvisited node or the depot (lookahead penalty)
       This avoids picking nodes that lead to long detours.
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
    
    # Immediate distance from current node to each feasible candidate
    immediate = distance_matrix[current_node, feasible_arr]
    
    # Lookahead: for each candidate, find the minimum distance to any other
    # feasible unvisited node or the depot
    # We want to avoid candidates that are far from everything else (dead-ends)
    # Lookahead cost = min distance from candidate to (other feasible nodes ∪ {depot})
    
    # Build a small matrix: distances from each feasible candidate to all feasible candidates + depot
    # candidates: feasible_arr, depot
    all_targets = np.concatenate([feasible_arr, [depot]])
    # distance_matrix[candidate, target] for all pairs
    # Shape: (len(feasible), len(feasible)+1)
    lookahead_dists = distance_matrix[feasible_arr][:, all_targets]
    
    # For each candidate i, the minimum distance to any target (including itself is 0, so exclude self)
    # Actually, we want min distance to OTHER nodes or depot
    # Set diagonal to inf to exclude self
    n_feas = len(feasible_arr)
    # lookahead_dists shape: (n_feas, n_feas+1)
    # Column j (0..n_feas-1) corresponds to feasible_arr[j], column n_feas corresponds to depot
    # For row i, we want min over all columns except column i (self)
    # Create a copy and set self to inf
    lookahead_min = lookahead_dists.copy()
    for i in range(n_feas):
        lookahead_min[i, i] = np.inf
    # Now min over axis 1
    lookahead_cost = np.min(lookahead_min, axis=1)
    
    # Composite score: immediate distance + alpha * lookahead_cost
    # alpha controls the weight of lookahead; smaller alpha favors immediate proximity
    alpha = 0.5
    scores = immediate + alpha * lookahead_cost
    
    # Pick the candidate with the lowest score
    best_idx = int(np.argmin(scores))
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
