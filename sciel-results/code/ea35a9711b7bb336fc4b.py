"""Capacitated facility location: portfolio greedy seeds + optimized local search with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_seed(demands, capacities, open_cost, assign_cost, strategy='demand_desc'):
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac, dtype=float)

    if strategy == 'demand_desc':
        total_cost = assign_cost.sum(axis=1)
        order = np.lexsort((total_cost, -demands))
    elif strategy == 'demand_asc':
        order = np.argsort(demands)
    elif strategy == 'random':
        order = np.random.permutation(n_cust)
    elif strategy == 'cheapest_first':
        min_costs = np.min(assign_cost, axis=1)
        order = np.argsort(min_costs)
    else:
        total_cost = assign_cost.sum(axis=1)
        order = np.lexsort((total_cost, -demands))

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
            f = int(np.argmin(load))
            load[f] += demands[i]
            result[i] = f
    return result, load


def _objective(result, load, open_cost, assign_cost, demands, capacities):
    used = load > 0
    open_part = open_cost[used].sum()
    assign_part = assign_cost[np.arange(len(result)), result].sum()
    return open_part + assign_part


def _local_search(demands, capacities, open_cost, assign_cost, result, load, deadline):
    n_cust = len(demands)
    n_fac = len(capacities)
    fac_order = np.argsort(assign_cost, axis=1)

    cur_obj = _objective(result, load, open_cost, assign_cost, demands, capacities)

    improved = True
    no_improve_pass = 0
    while improved and time.time() < deadline:
        improved = False
        # Single reassignment pass
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
                if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                    continue
                da = assign_cost[i, f_new] - base_assign
                do = 0.0
                if load[f_old] - demands[i] <= 1e-9:
                    do -= open_cost[f_old]
                if load[f_new] <= 1e-9:
                    do += open_cost[f_new]
                gain = -(da + do)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_f = f_new
            if best_f >= 0:
                load[f_old] -= demands[i]
                load[best_f] += demands[i]
                result[i] = best_f
                cur_obj -= best_gain
                improved = True

        # Swap pass
        if time.time() < deadline:
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
                    if load[fi] - di + dj > capacities[fi] + 1e-9:
                        continue
                    if load[fj] - dj + di > capacities[fj] + 1e-9:
                        continue
                    da = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                    do = 0.0
                    if load[fi] - di + dj <= 1e-9 and load[fi] > 1e-9:
                        do -= open_cost[fi]
                    if load[fj] - dj + di <= 1e-9 and load[fj] > 1e-9:
                        do -= open_cost[fj]
                    gain = -(da + do)
                    if gain > 1e-12:
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
    deadline = t0 + 140.0  # safety margin under 150s

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    # Portfolio of initializations
    strategies = ['demand_desc', 'demand_asc', 'random', 'cheapest_first']
    best_cost = float('inf')
    best_result = None
    best_load = None

    for strategy in strategies:
        if time.time() > deadline - 20.0:
            break

        result, load = _greedy_seed(demands, capacities, open_cost, assign_cost, strategy)
        current_cost = _objective(result, load, open_cost, assign_cost, demands, capacities)

        # Local search
        result, load, current_cost = _local_search(demands, capacities, open_cost, assign_cost, result, load, deadline - 10.0)

        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()
            best_load = load.copy()

    if best_result is None:
        best_result, best_load = _greedy_seed(demands, capacities, open_cost, assign_cost, 'demand_desc')

    # Final validation: ensure capacities respected and indices in range
    result = best_result
    load_check = np.zeros(n_fac)
    for i in range(n_cust):
        f = result[i]
        if f < 0 or f >= n_fac:
            f = int(np.argmin(load_check))
            result[i] = f
        load_check[f] += demands[i]

    # If any capacity violated, do a quick repair
    for f in range(n_fac):
        if load_check[f] > capacities[f] + 1e-9:
            excess = load_check[f] - capacities[f]
            custs_at_f = np.where(result == f)[0]
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
