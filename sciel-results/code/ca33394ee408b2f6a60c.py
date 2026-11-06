"""Capacitated vehicle routing – combined regret + savings lookahead selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 120.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Safety: return depot if we're close to the deadline
    if _elapsed() > _DEADLINE - 5.0:
        return int(depot)

    unvisited_nodes = np.asarray(unvisited_nodes, dtype=int)
    if len(unvisited_nodes) == 0:
        return int(depot)

    n = len(unvisited_nodes)
    dmat = np.asarray(distance_matrix)

    # Feasible nodes: demand fits remaining capacity
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return int(depot)

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Precompute distances from current node to all feasible nodes
    dists = dmat[current_node, feasible_nodes]

    # Compute Clarke-Wright savings for each feasible node
    # savings(j) = d(current, depot) + d(depot, j) - d(current, j)
    d_current_depot = dmat[current_node, depot]
    d_depot_feasible = dmat[depot, feasible_nodes]
    savings = d_current_depot + d_depot_feasible - dists

    # Compute 1-step lookahead: for each feasible node j, estimate the cost
    # of the next step after visiting j (min distance to another feasible node
    # that can fit in remaining capacity, or to depot if none)
    lookahead = np.zeros(len(feasible_nodes))
    for idx, j in enumerate(feasible_nodes):
        remaining_cap = rest_capacity - demands[j]
        # Nodes that can be visited after j
        cap_mask = demands[unvisited_nodes] <= remaining_cap
        not_j_mask = unvisited_nodes != j
        next_feasible_mask = cap_mask & not_j_mask
        next_feasible_nodes = unvisited_nodes[next_feasible_mask]

        if len(next_feasible_nodes) > 0:
            lookahead[idx] = np.min(dmat[j, next_feasible_nodes])
        else:
            lookahead[idx] = dmat[j, depot]

    # Compute regret-2 based on a combined cost metric
    # cost(j) = dists[j] + lookahead[j] - savings[j]
    # This combines immediate distance, future cost, and savings benefit
    combined_cost = dists + lookahead - 1.0 * savings

    # Regret-2: for each node, regret = min_cost_excluding_j - cost_j
    # We want to maximize regret (nodes that are uniquely good)
    sorted_cost_idx = np.argsort(combined_cost)
    min1_val = combined_cost[sorted_cost_idx[0]]
    min2_val = combined_cost[sorted_cost_idx[1]]
    min1_pos = sorted_cost_idx[0]

    # For each node, find the best alternative cost
    best_other = np.where(
        np.arange(len(feasible_nodes)) == min1_pos,
        min2_val,
        min1_val
    )
    regret = best_other - combined_cost

    # Also compute a simple distance-based regret as a secondary signal
    sorted_dist_idx = np.argsort(dists)
    dist_min1 = dists[sorted_dist_idx[0]]
    dist_min2 = dists[sorted_dist_idx[1]]
    dist_min1_pos = sorted_dist_idx[0]

    dist_best_other = np.where(
        np.arange(len(feasible_nodes)) == dist_min1_pos,
        dist_min2,
        dist_min1
    )
    dist_regret = dist_best_other - dists

    # Combine signals:
    # - Primary: combined regret (savings + lookahead aware)
    # - Secondary: distance regret (uniqueness of position)
    # - Tertiary: slight preference for lower demand to preserve capacity flexibility
    # - Quaternary: slight penalty for nodes far from depot (harder to close route)

    demand_penalty = 0.02 * demands[feasible_nodes]
    depot_dist_penalty = 0.03 * dmat[feasible_nodes, depot]

    # Normalize regret scores to [0, 1] range for stable combination
    regret_max = np.max(regret)
    regret_min = np.min(regret)
    if regret_max > regret_min:
        regret_norm = (regret - regret_min) / (regret_max - regret_min)
    else:
        regret_norm = np.ones_like(regret)

    dist_regret_max = np.max(dist_regret)
    dist_regret_min = np.min(dist_regret)
    if dist_regret_max > dist_regret_min:
        dist_regret_norm = (dist_regret - dist_regret_min) / (dist_regret_max - dist_regret_min)
    else:
        dist_regret_norm = np.ones_like(dist_regret)

    # Weighted combination
    score = (0.6 * regret_norm + 0.3 * dist_regret_norm) - demand_penalty - depot_dist_penalty

    best_idx = np.argmax(score)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
