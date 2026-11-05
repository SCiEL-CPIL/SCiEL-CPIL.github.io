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
    # after picking it. We use a smooth surrogate for the minimum distance to
    # other feasible nodes or the depot.
    
    # Build a small (k x k) sub-matrix among feasible nodes
    sub = distance_matrix[np.ix_(feas, feas)]
    
    # Distance from each feasible node to the depot
    d_depot = distance_matrix[feas, depot]
    
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(sub, np.inf)
    
    # For each candidate i, the "next step" cost is min(min distance to other feasible, distance to depot)
    # We use a soft-min (log-sum-exp) for smoothness, but since we just need argmin,
    # hard min is fine and faster. Let's use hard min for speed.
    nearest_other = sub.min(axis=1)
    
    # The true next-step cost for candidate i is min(nearest_other[i], d_depot[i])
    # This represents: after visiting i, we either go to another feasible node or return to depot.
    next_step_cost = np.minimum(nearest_other, d_depot)
    
    # Smooth surrogate: score = d_cur + alpha * next_step_cost
    # We want to balance immediate distance with future cost.
    # A higher alpha means we care more about the future.
    # Let's try alpha = 0.5 for a stronger lookahead.
    alpha = 0.5
    scores = d_cur + alpha * next_step_cost
    
    idx = int(np.argmin(scores))
    return int(feas[idx])
# EVOLVE-BLOCK-END
