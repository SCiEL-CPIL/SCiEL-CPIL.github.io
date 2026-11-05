"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if not _time_ok():
        if len(unvisited_nodes) == 0:
            return int(depot)
        unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
        feas_mask = demands[unvisited_arr] <= rest_capacity
        if not np.any(feas_mask):
            return int(depot)
        feas = unvisited_arr[feas_mask]
        return int(feas[np.argmin(distance_matrix[current_node, feas])])

    if len(unvisited_nodes) == 0:
        return int(depot)

    # Convert to numpy array
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)

    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas_nodes = unvisited_arr[feas_mask]

    if len(feas_nodes) == 1:
        return int(feas_nodes[0])

    # Distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]

    # Savings heuristic:
    # Saving = d[current, depot] + d[depot, c] - d[current, c]
    # We want to maximize this.
    savings = d_cur_dep + d_dep_feas - d_cur_feas

    # Structural change: use a composite score that balances:
    # 1. Savings (maximize) - encourages connecting nodes in a chain
    # 2. Distance from current (minimize) - keeps route compact, avoids zig-zag
    # 3. Distance to depot when capacity is tight (minimize) - prepares for route closure
    #
    # The key insight: instead of pure savings, we use a weighted combination
    # that adapts based on remaining capacity. When capacity is tight, we
    # should prefer nodes close to the depot to minimize the closing cost.
    #
    # Score = -savings + w1 * d_cur_feas + w2 * tightness * d_dep_feas
    # where tightness = 1 - rest_capacity / max_feasible_demand
    #
    # This is a structural change from the pure savings approach: it adds
    # two additional terms that adapt the heuristic based on the state of
    # the route (how tight the capacity is).

    # Compute tightness: how close we are to filling the vehicle
    feas_demands = demands[feas_nodes]
    max_demand = np.max(feas_demands) if len(feas_demands) > 0 else 1
    if max_demand > 0:
        tightness = 1.0 - (rest_capacity / max_demand)
        tightness = np.clip(tightness, 0.0, 1.0)
    else:
        tightness = 0.0

    # Weights:
    # - savings is the primary signal (weight 1.0)
    # - distance to current is a secondary signal (weight 0.15)
    # - distance to depot is a tertiary signal, scaled by tightness (weight 0.4 * tightness)
    #
    # The tightness term ensures that when we're close to capacity limit,
    # we prefer nodes that are close to the depot, which will reduce the
    # cost of closing the route.

    w1 = 0.15
    w2 = 0.4

    scores = -savings + w1 * d_cur_feas + w2 * tightness * d_dep_feas

    best_idx = np.argmin(scores)
    best_node = feas_nodes[best_idx]

    return int(best_node)
# EVOLVE-BLOCK-END
