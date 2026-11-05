"""Rewritten program for Capacitated Vehicle Routing; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 148.0

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Safety: if time is running out, close the route.
    if not _time_ok():
        return int(depot)

    if len(unvisited_nodes) == 0:
        return int(depot)

    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)

    # Capacity feasibility: node's demand must fit in remaining capacity.
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas_nodes = unvisited_arr[feas_mask]
    n_feas = feas_nodes.shape[0]

    # Distance slices (row vectors).
    d_cur = distance_matrix[current_node, feas_nodes]   # current -> candidate
    d_dep = distance_matrix[depot, feas_nodes]          # depot -> candidate

    # Forward look: for each candidate c, find the cheapest feasible successor s
    # (s in feas, s != c, demand(s) <= rest_capacity - demand(c)), then estimate
    # the extra route cost contributed by choosing c now:
    #   leg1 = d[cur, c]
    #   leg2 = d[c, best_succ]        (or d[c, depot] if none)
    #   leg3 = d[best_succ, depot]    (or 0 if none)
    # and a pull-back term that rewards nodes that are "far from depot" relative
    # to "near current" (classic savings), but bounded.

    # Precompute full sub-matrix among feasible nodes.
    D = distance_matrix[np.ix_(feas_nodes, feas_nodes)]   # n_feas x n_feas
    dem = demands[feas_nodes]
    rem = rest_capacity

    # best successor distance to depot and from candidate.
    leg2 = np.empty(n_feas, dtype=float)
    leg3 = np.empty(n_feas, dtype=float)

    # Soft-min temperature: small enough to approximate min, large enough to
    # incorporate second-best options when they are close in cost.
    beta = 4.0  # 1/temperature; higher = sharper min

    for i in range(n_feas):
        ci = dem[i]
        can_rem = rem - ci
        # feasible successors: index j != i with dem[j] <= can_rem
        if can_rem <= 0:
            # no successor fits; must go to depot next
            leg2[i] = distance_matrix[feas_nodes[i], depot]
            leg3[i] = 0.0
            continue
        succ_mask = (dem <= can_rem)
        succ_mask[i] = False
        if np.any(succ_mask):
            row = D[i, succ_mask]
            succ_dep = d_dep[succ_mask]
            # cost if we go to successor j: row[j] + d[depot, j]
            combined = row + succ_dep
            # Soft-min: -log(sum(exp(-beta * combined))) / beta
            # This gives a smooth estimate of the min that also accounts
            # for the distribution of close alternatives.
            neg_beta_c = -beta * combined
            max_neg = np.max(neg_beta_c)
            log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_c - max_neg)))
            soft_min_val = -log_sum / beta

            # For leg2/leg3, use the argmin (hard) for the actual path cost,
            # but blend with the soft-min to get a smoother total estimate.
            jbest = np.argmin(combined)
            succ_indices = np.where(succ_mask)[0]
            jb = succ_indices[jbest]
            hard_leg2 = row[jbest]
            hard_leg3 = succ_dep[jbest]

            # Blend: 70% hard (preserves path structure), 30% soft (smooths)
            # The soft-min gives a single scalar; we approximate the split
            # by scaling the hard legs by the ratio soft/hard total.
            hard_total = hard_leg2 + hard_leg3
            if hard_total > 1e-9:
                ratio = soft_min_val / hard_total
                # Clamp ratio to [0.8, 1.0] to avoid over-discounting
                ratio = np.clip(ratio, 0.8, 1.0)
                leg2[i] = hard_leg2 * ratio
                leg3[i] = hard_leg3 * ratio
            else:
                leg2[i] = hard_leg2
                leg3[i] = hard_leg3
        else:
            leg2[i] = distance_matrix[feas_nodes[i], depot]
            leg3[i] = 0.0

    # Estimated marginal cost of choosing candidate i now:
    #   total = leg1 + leg2 + leg3
    #   minus a pull-back reward: nodes that are far from depot but close to current
    #   are good (they save a separate depot detour). Pull-back = alpha * (d_dep - d_cur)
    #   bounded so it doesn't dominate the forward look.
    alpha = 0.30
    pull = alpha * np.clip(d_dep - d_cur, -1.0, None)

    total = d_cur + leg2 + leg3 - pull

    # Tie-break: prefer smaller immediate distance.
    order = np.lexsort((d_cur, total))
    best_idx = order[0]
    best_node = feas_nodes[best_idx]

    return int(best_node)
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
