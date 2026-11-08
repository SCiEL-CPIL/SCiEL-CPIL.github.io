"""Capacitated facility location: greedy seed + local-search refinement with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_seed(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=float)

    # Process largest demand first; break ties by smaller total assign cost (more constrained).
    total_cost = assign_cost.sum(axis=1)
    order = np.lexsort((total_cost, -demands))

    # Precompute facility ordering per customer by cost (vectorized once).
    # We'll just sort per customer on the fly but cache the argorder.
    fac_order = np.argsort(assign_cost, axis=1)

    for i in order:
        placed = False
        for f in fac_order[i]:
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # Fallback: put in least-loaded facility (should not happen if feasible)
            f = int(np.argmin(load))
            load[f] += demands[i]
            result[i] = f
    return result, load


def _objective(result, load, open_cost, assign_cost, demands, capacities):
    # open cost for facilities with >=1 customer
    used = load > 0
    open_part = open_cost[used].sum()
    assign_part = assign_cost[np.arange(len(result)), result].sum()
    return open_part + assign_part


def _local_search(demands, capacities, open_cost, assign_cost, result, load, deadline):
    n_cust = len(demands)
    n_fac = len(capacities)
    fac_order = np.argsort(assign_cost, axis=1)

    # Current objective
    cur_obj = _objective(result, load, open_cost, assign_cost, demands, capacities)

    # Track which facilities are "open" (have load > 0)
    # A move that empties a facility saves its open_cost.

    improved = True
    no_improve_pass = 0
    while improved and time.time() < deadline:
        improved = False
        # Single reassignment pass: for each customer, try moving to a cheaper facility
        # where capacity allows, and the net objective decreases.
        # Net change for moving customer i from f_old to f_new:
        #   assign cost delta = assign_cost[i, f_new] - assign_cost[i, f_old]
        #   open cost delta: if load[f_old] - demands[i] <= 0 and load[f_old] > 0: -open_cost[f_old]
        #                    if load[f_new] <= 0 and f_new != f_old: +open_cost[f_new]
        for i in range(n_cust):
            if time.time() >= deadline:
                break
            f_old = result[i]
            if f_old < 0:
                continue
            best_gain = 0.0
            best_f = -1
            base_assign = assign_cost[i, f_old]
            for f_new in fac_order[i]:
                if f_new == f_old:
                    continue
                # capacity check
                if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                    continue
                da = assign_cost[i, f_new] - base_assign
                do = 0.0
                # old facility empties?
                if load[f_old] - demands[i] <= 1e-9:
                    do -= open_cost[f_old]
                # new facility opens?
                if load[f_new] <= 1e-9:
                    do += open_cost[f_new]
                gain = -(da + do)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_f = f_new
            if best_f >= 0:
                # apply
                load[f_old] -= demands[i]
                load[best_f] += demands[i]
                result[i] = best_f
                cur_obj -= best_gain
                improved = True

        # Swap pass: two customers i, j with i<j, swap facilities if improves.
        if time.time() < deadline:
            # Limit swaps to keep it fast; do a limited number of random-ish pairs
            # Actually do a structured pass over pairs but only consider promising ones.
            # For speed, iterate over customers and try swapping with a small set.
            # We'll do a full O(n^2) but with early capacity/cost pruning, bounded by deadline.
            for i in range(n_cust):
                if time.time() >= deadline:
                    break
                fi = result[i]
                if fi < 0:
                    continue
                di = demands[i]
                for j in range(i + 1, n_cust):
                    if time.time() >= deadline:
                        break
                    fj = result[j]
                    if fj < 0 or fj == fi:
                        continue
                    dj = demands[j]
                    # after swap: i goes to fj, j goes to fi
                    # capacity checks
                    # load at fi after: load[fi] - di + dj ; must be <= cap[fi]
                    # load at fj after: load[fj] - dj + di ; must be <= cap[fj]
                    if load[fi] - di + dj > capacities[fi] + 1e-9:
                        continue
                    if load[fj] - dj + di > capacities[fj] + 1e-9:
                        continue
                    # assign cost delta
                    da = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                    # open cost delta
                    do = 0.0
                    # fi empties?
                    if load[fi] - di + dj <= 1e-9 and load[fi] > 1e-9:
                        do -= open_cost[fi]
                    # fj empties?
                    if load[fj] - dj + di <= 1e-9 and load[fj] > 1e-9:
                        do -= open_cost[fj]
                    # fi opens? (it was open since fi had i)
                    # fj opens? (it was open since fj had j)
                    # No new opens in a swap (both were open before)
                    gain = -(da + do)
                    if gain > 1e-12:
                        # apply swap
                        load[fi] = load[fi] - di + dj
                        load[fj] = load[fj] - dj + di
                        result[i], result[j] = fj, fi
                        cur_obj -= gain
                        improved = True

        if not improved:
            no_improve_pass += 1
            if no_improve_pass >= 1:
                break
        else:
            no_improve_pass = 0

    return result, load, cur_obj


def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s / 150s

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    result, load = _greedy_seed(demands, capacities, open_cost, assign_cost)

    # Validate seed
    if np.any(result < 0) or np.any(result >= len(capacities)):
        # Should not happen; fallback
        result = np.zeros(len(demands), dtype=int)
        load = np.zeros(len(capacities))
        for i in range(len(demands)):
            load[i % len(capacities)] += demands[i]
            result[i] = i % len(capacities)

    result, load, obj = _local_search(demands, capacities, open_cost, assign_cost, result, load, deadline)

    # Final validation: ensure capacities respected and indices in range
    load_check = np.zeros(len(capacities))
    for i in range(len(result)):
        f = result[i]
        if f < 0 or f >= len(capacities):
            # repair
            f = int(np.argmin(load_check))
            result[i] = f
        load_check[f] += demands[i]

    # If any capacity violated, do a quick repair: move overflowing customers
    for f in range(len(capacities)):
        if load_check[f] > capacities[f] + 1e-9:
            # move some customers from f to others
            excess = load_check[f] - capacities[f]
            custs_at_f = np.where(result == f)[0]
            # move largest first
            custs_at_f = custs_at_f[np.argsort(-demands[custs_at_f])]
            for i in custs_at_f:
                if excess <= 1e-9:
                    break
                for f2 in np.argsort(assign_cost[i]):
                    if f2 == f:
                        continue
                    if load_check[f2] + demands[i] <= capacities[f2] + 1e-9:
                        load_check[f] -= demands[i]
                        load_check[f2] += demands[i]
                        result[i] = f2
                        excess -= demands[i]
                        break

    return result.astype(int)
# EVOLVE-BLOCK-END
