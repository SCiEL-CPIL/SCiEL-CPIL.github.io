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

    # Compute 2-step lookahead: for each feasible node j, estimate the cost
    # of the next two steps after visiting j.
    # Step 1: min distance from j to another feasible node k (capacity-feasible)
    # Step 2: min distance from k to another feasible node l, or to depot
    lookahead = np.zeros(len(feasible_nodes))
    n_feasible = len(feasible_nodes)
    for idx, j in enumerate(feasible_nodes):
        remaining_cap = rest_capacity - demands[j]
        # Nodes that can be visited after j
        cap_mask = demands[unvisited_nodes] <= remaining_cap
        not_j_mask = unvisited_nodes != j
        next_feasible_mask = cap_mask & not_j_mask
        next_feasible_nodes = unvisited_nodes[next_feasible_mask]

        if len(next_feasible_nodes) == 0:
            # Must go back to depot
            lookahead[idx] = dmat[j, depot]
        else:
            # Find the best next node k minimizing d(j,k) + min(d(k, l) for l feasible after k, or d(k, depot))
            best_second_cost = np.inf
            for k in next_feasible_nodes:
                dk = dmat[j, k]
                # After visiting k, what's the min cost to continue?
                remaining_cap2 = remaining_cap - demands[k]
                cap_mask2 = demands[unvisited_nodes] <= remaining_cap2
                not_jk_mask = (unvisited_nodes != j) & (unvisited_nodes != k)
                next2_mask = cap_mask2 & not_jk_mask
                next2_nodes = unvisited_nodes[next2_mask]
                if len(next2_nodes) > 0:
                    # Can visit another node after k
                    min_next2 = np.min(dmat[k, next2_nodes])
                else:
                    # Must return to depot after k
                    min_next2 = dmat[k, depot]
                total = dk + min_next2
                if total < best_second_cost:
                    best_second_cost = total
            lookahead[idx] = best_second_cost

    # Compute regret-2 based on a combined cost metric
    # cost(j) = dists[j] + 0.5 * lookahead[j] - 1.5 * savings[j]
    # Increased savings weight (1.5) based on finding that pure savings outperformed hybrid
    # Reduced lookahead weight (0.5) since 2-step is already more informative
    combined_cost = dists + 0.5 * lookahead - 1.5 * savings

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
    # - Primary: combined regret (savings + 2-step lookahead aware)
    # - Secondary: savings-based regret (uniqueness of savings opportunity)
    # - Tertiary: slight preference for lower demand to preserve capacity flexibility

    # Savings-based regret: how unique is this node's savings value?
    sorted_savings_idx = np.argsort(-savings)  # descending by savings
    sav_max1 = savings[sorted_savings_idx[0]]
    sav_max2 = savings[sorted_savings_idx[1]]
    sav_max1_pos = sorted_savings_idx[0]

    sav_best_other = np.where(
        np.arange(n_feasible) == sav_max1_pos,
        sav_max2,
        sav_max1
    )
    sav_regret = sav_best_other - savings

    demand_penalty = 0.02 * demands[feasible_nodes]

    # Normalize regret scores to [0, 1] range for stable combination
    regret_max = np.max(regret)
    regret_min = np.min(regret)
    if regret_max > regret_min:
        regret_norm = (regret - regret_min) / (regret_max - regret_min)
    else:
        regret_norm = np.ones_like(regret)

    sav_regret_max = np.max(sav_regret)
    sav_regret_min = np.min(sav_regret)
    if sav_regret_max > sav_regret_min:
        sav_regret_norm = (sav_regret - sav_regret_min) / (sav_regret_max - sav_regret_min)
    else:
        sav_regret_norm = np.ones_like(sav_regret)

    # Weighted combination: emphasize savings regret more (measured better)
    score = (0.5 * regret_norm + 0.4 * sav_regret_norm) - demand_penalty

    best_idx = np.argmax(score)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
