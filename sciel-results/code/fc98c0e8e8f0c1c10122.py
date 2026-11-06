"""Spatial-cluster consolidation heuristic for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import time
import math

import numpy as np

# --- explicit deadline guard: 150s wall-clock limit, stop with safety margin ---
_T0 = time.time()
_DEADLINE = 126.0  # seconds of wall-clock budget (150s - 24s margin)
_TIME_LIMITED = True


def _now():
    return time.time()


def _time_ok():
    return (not _TIME_LIMITED) or (_now() - _T0) < _DEADLINE


def _cluster_score(node, current_node, feasible, demands, dist):
    """Score a candidate by localized demand-density consolidation.

    High score = the node's Voronoi-region (nearest feasible customers) carries
    a lot of demand close by, so the route can exhaust capacity inside a tight
    geographic cluster. We blend:
      - demand density in the local region (favors capacity exhaustion locally)
      - proximity of the cluster to the current position (favors compact routes)
    """
    if len(feasible) == 0:
        return 0.0

    # distances from candidate to all feasible customers (its Voronoi region)
    d_to_feas = dist[node, feasible]
    # local demand density: total feasible demand weighted by proximity
    # (closer customers contribute more to "local" consolidation)
    w = 1.0 / (d_to_feas + 1.0)
    local_density = float(np.sum(demands[feasible] * w))

    # how far is the densest local demand from our current position?
    # (penalize clusters that are far from where we are)
    d_cur_to_feas = dist[current_node, feasible]
    # effective cluster distance: demand-weighted mean distance to local cluster
    eff_dist = float(np.sum(d_cur_to_feas * w) / (np.sum(w) + 1e-9))

    # consolidation value: dense local demand, close to us
    # normalize-ish: density grows with many close customers
    return local_density / (eff_dist + 1.0)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands,
                     distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.asarray(unvisited_nodes)
    dist = np.asarray(distance_matrix)
    dem = np.asarray(demands)

    # capacity-feasible customers only
    feasible_mask = dem[unvisited] <= rest_capacity
    feasible = unvisited[feasible_mask]

    if len(feasible) == 0:
        return depot

    # --- base cluster-consolidation score for every feasible candidate ---
    scores = np.empty(len(feasible))
    for i, node in enumerate(feasible):
        scores[i] = _cluster_score(node, current_node, feasible, dem, dist)

    # --- time-bounded local lookahead to refine the pick ---
    # Consider the top-K cluster candidates and do a 1-step lookahead:
    # for each, estimate the cost of the best next move after it.
    K = min(len(feasible), 8)
    order = np.argsort(-scores)[:K]

    best_node = int(feasible[order[0]])
    best_total = float('inf')

    for idx in order:
        if not _time_ok():
            break
        node = feasible[idx]
        # after taking node, remaining capacity
        rem = rest_capacity - dem[node]
        # cost to reach node from current
        cost_in = dist[current_node, node]
        # best next move from node (nearest feasible among remaining)
        if len(feasible) > 1:
            others = np.delete(feasible, np.where(feasible == node)[0][0])
            feasible_others = others[dem[others] <= rem]
            if len(feasible_others) > 0:
                cost_out = float(np.min(dist[node, feasible_others]))
            else:
                cost_out = float(dist[node, depot])  # must return to depot
        else:
            cost_out = float(dist[node, depot])

        total = cost_in + cost_out
        # combine: prefer high cluster score, low lookahead cost
        combined = scores[idx] - 0.05 * total
        if combined > best_total or (combined == best_total and total < best_total):
            best_total = combined
            best_node = int(node)

    return int(best_node)
# EVOLVE-BLOCK-END
