"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 148.0
_SAFETY_MARGIN = 2.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated vehicle routing:
    Pick the next customer to visit, or return to depot (0) to close the route.
    
    Strategy: Multi-criteria scoring with lookahead.
    1. Filter unvisited nodes that fit within rest_capacity.
    2. If none fit, return depot.
    3. Score each feasible candidate using a combination of:
       - Immediate distance from current_node to candidate
       - 2-step lookahead: min distance from candidate to other feasible nodes or depot
       - Capacity utilization penalty (prefer filling the truck)
    4. Return the candidate with the lowest score (best choice).
    """
    if _time_exceeded():
        # Fallback: nearest feasible node
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return 0
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        return int(feasible_arr[np.argmin(distance_matrix[current_node, feasible_arr])])
    
    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return 0  # Return to depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    n_feasible = len(feasible_arr)
    
    # Compute immediate distances from current node to each feasible candidate
    dists = distance_matrix[current_node, feasible_arr]
    
    # If no unvisited nodes remain after this one, just pick nearest
    if n_feasible == len(unvisited_nodes) and len(unvisited_nodes) <= 2:
        # Simple case: just pick nearest
        return int(feasible_arr[np.argmin(dists)])
    
    # Compute lookahead scores
    # For each candidate i, compute the minimum distance to any other feasible node or depot
    # This penalizes candidates that are "dead ends"
    
    # Precompute distance matrix among feasible nodes and to depot
    # dist_feasible[i][j] = distance from feasible[i] to feasible[j]
    # dist_to_depot[i] = distance from feasible[i] to depot
    
    # To be efficient, use the original distance matrix
    # For each candidate i, lookahead = min(
    #     min_j dist(feasible[i], feasible[j]) for j != i,
    #     dist(feasible[i], depot)
    # )
    
    # We want to minimize total route length, so we prefer candidates that are close now
    # AND close to other remaining nodes (to avoid long detours)
    
    # Score = dist_current_to_candidate + alpha * lookahead_penalty
    # where lookahead_penalty encourages picking nodes that keep options open
    
    # Compute min distance from each feasible node to any other feasible node or depot
    # This is a vectorized operation
    
    # dist_feasible_to_all: shape (n_feasible, n_feasible)
    # For each row i, we want min over all j of dist(feasible[i], feasible[j]), 
    # but we can include depot as an option
    
    # Let's create an augmented matrix: feasible nodes + depot
    # Actually, simpler: for each candidate i, 
    #   min_next = min(min_{j in feasible, j != i} dist(feasible[i], feasible[j]), dist(feasible[i], depot))
    
    # Vectorized approach:
    # Get the submatrix of distances between feasible nodes
    feas_dist = distance_matrix[np.ix_(feasible_arr, feasible_arr)]  # (n, n)
    dist_to_depot = distance_matrix[feasible_arr, depot]  # (n,)
    
    # For each i, find min distance to other feasible nodes (excluding self)
    # Set diagonal to inf, then take min along axis 1
    np.fill_diagonal(feas_dist, np.inf)
    min_to_other_feasible = np.min(feas_dist, axis=1)  # (n,)
    
    # The lookahead for each candidate is the min of min_to_other_feasible and dist_to_depot
    lookahead = np.minimum(min_to_other_feasible, dist_to_depot)
    
    # Also consider: if this is the last feasible node, lookahead should be dist to depot
    # But the min already handles that since if no other feasible nodes, min_to_other_feasible is inf
    
    # Now score: we want to minimize dists + beta * lookahead
    # But we also want to consider capacity: if rest_capacity is tight, prefer higher demand nodes
    # to fill the truck
    
    # Capacity utilization bonus: prefer nodes with higher demand relative to rest_capacity
    # This helps fill trucks more completely
    demand_ratio = demands[feasible_arr] / max(rest_capacity, 1)
    
    # Combined score:
    # Primary: immediate distance
    # Secondary: lookahead (penalize dead-ends)
    # Tertiary: capacity utilization (prefer filling the truck)
    
    # Normalize scores to comparable scales
    # Use a weighted sum
    
    alpha = 0.3  # Weight for lookahead
    gamma = 0.1  # Weight for capacity utilization (negative, so higher demand is better)
    
    # Score = dists + alpha * lookahead - gamma * demand_ratio * (some_scale)
    # Let's use: score = dists + alpha * lookahead - gamma * demand_ratio
    # But demand_ratio is in [0,1], so gamma should be small relative to distances
    
    # Actually, let's think about this differently.
    # The main goal is to minimize total route length.
    # A good heuristic is to minimize: dist(current, candidate) + lambda * min_dist(candidate, remaining_or_depot)
    # This is similar to the "regret" or "lookahead" approach.
    
    # Let's also add a small penalty for low capacity utilization to encourage fuller trucks
    # But this is secondary to route length.
    
    scores = dists + alpha * lookahead - gamma * demand_ratio
    
    # Find the best candidate
    best_idx = np.argmin(scores)
    
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
