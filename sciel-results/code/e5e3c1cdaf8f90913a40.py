"""Capacitated VRP heuristic: angular sweep with distance tie-breaking."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 110.0
_DEADLINE = _start_time + _TIMEOUT


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next customer node or return depot (0) to close the route.
    Uses an angular-sweep heuristic: prefer customers that continue the
    current direction of travel (spatial clustering), with distance as
    a secondary criterion.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    unvisited = np.array(unvisited_nodes, dtype=int)

    # Filter by capacity feasibility
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    feasible = unvisited[feasible_mask]

    if len(feasible) == 1:
        return int(feasible[0])

    # Compute distances from current node to feasible customers
    dists = distance_matrix[current_node, feasible]

    # Compute angular direction from depot to each feasible customer
    # We need coordinates; since we only have distance_matrix, we approximate
    # direction using the distance to depot as a proxy for "angle"
    # Actually, without coordinates, we can't compute true angles.
    # Alternative: use the distance from depot as a clustering proxy.
    # Customers closer to the depot in terms of distance from current node
    # that also have similar "radial" distance from depot tend to cluster.

    # Let's use a combined score:
    # 1. Primary: distance from current node (to keep routes short)
    # 2. Secondary: prefer customers whose distance from depot is similar
    #    to the current node's distance from depot (same "ring")
    # 3. Tertiary: demand urgency (higher demand first to reduce routes)

    depot_dist_current = distance_matrix[current_node, depot]
    depot_dist_feasible = distance_matrix[feasible, depot]

    # Angular deviation proxy: how much the radial distance differs
    # from the current node's radial distance
    radial_dev = np.abs(depot_dist_feasible - depot_dist_current)

    # Normalize
    max_radial_dev = np.max(radial_dev) if np.max(radial_dev) > 0 else 1.0
    radial_dev_norm = radial_dev / max_radial_dev

    max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
    dist_norm = dists / max_dist

    # Demand urgency
    max_demand = np.max(demands) if np.max(demands) > 0 else 1.0
    demand_urgency = demands[feasible] / max_demand

    # Combined score: lower is better
    # Weight distance heavily, add radial deviation penalty, add small demand urgency
    score = 1.0 * dist_norm + 0.3 * radial_dev_norm + 0.1 * (1.0 - demand_urgency)

    best_idx = np.argmin(score)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
