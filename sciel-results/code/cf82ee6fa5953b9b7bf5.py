"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # safety margin before 150s kill


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    # Feasible: demand fits in remaining capacity
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    k = len(feas)

    # Distance from current node to each feasible candidate
    d_cur = distance_matrix[current_node, feas]

    if k == 1:
        return int(feas[0])

    # Lookahead: for each candidate, estimate the cost of continuing the route
    # after picking it. We use a smooth surrogate (soft-min) for the minimum distance
    # to other feasible nodes or the depot.
    
    # Build a small (k x k) sub-matrix among feasible nodes
    sub = distance_matrix[np.ix_(feas, feas)].copy()
    
    # Distance from each feasible node to the depot
    d_depot = distance_matrix[feas, depot]
    
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(sub, np.inf)
    
    # Soft-min (log-sum-exp) of distances to other feasible nodes
    # This captures not just the nearest neighbor but the distribution of options
    # Temperature parameter controls how "soft" the min is
    temp = 0.1
    # Subtract max for numerical stability
    sub_shifted = sub - sub.max(axis=1, keepdims=True)
    soft_min_other = -temp * np.log(np.sum(np.exp(sub_shifted / temp), axis=1)) + sub.max(axis=1)
    
    # Soft-min of distance to depot and other feasible nodes
    # Combine into a single (k x (k+1)) matrix: [sub | d_depot]
    # But we can do it more efficiently:
    # soft_min_combined = soft_min of [sub, d_depot]
    # We'll create a (k x (k+1)) array
    combined = np.column_stack([sub, d_depot])
    combined_shifted = combined - combined.max(axis=1, keepdims=True)
    soft_min_combined = -temp * np.log(np.sum(np.exp(combined_shifted / temp), axis=1)) + combined.max(axis=1)
    
    # Score: balance immediate distance with expected future cost
    # Higher alpha means we care more about the future
    alpha = 0.3
    scores = d_cur + alpha * soft_min_combined
    
    idx = int(np.argmin(scores))
    return int(feas[idx])
# EVOLVE-BLOCK-END
