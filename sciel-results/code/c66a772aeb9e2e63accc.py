# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def _total_cost(result, demands, capacities, open_cost, assign_cost):
    n_fac = len(capacities)
    used = np.zeros(n_fac, dtype=bool)
    for f in result:
        used[f] = True
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    if np.any(load > capacities):
        return np.inf
    return float(np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(len(demands)), result]))


def _greedy(demands, capacities, open_cost, assign_cost, mode):
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    result = np.full(n_cust, -1, dtype=int)
    if mode == 'nearest':
        order = np.arange(n_cust)
    elif mode == 'largest':
        order = np.argsort(-demands)
    elif mode == 'effective':
        # effective cost: assignment cost adjusted by marginal open cost
        eff = assign_cost.copy()
        # add a small penalty spread of open cost to discourage opening many
        for f in range(n_fac):
            eff[:, f] += open_cost[f] / n_fac
        # sort customers by largest demand first within nearest-effective
        order = np.argsort(-demands)
    elif mode == 'random':
        order = np.random.permutation(n_cust)
    else:
        order = np.arange(n_cust)

    for i in order:
        if mode == 'effective':
            # recompute effective cost considering which facilities are open
            open_mask = load > 0
            eff_row = assign_cost[i].copy()
            # for closed facilities, add open_cost
            eff_row[~open_mask] += open_cost[~open_mask]
            cand = np.argsort(eff_row)
        else:
            cand = np.argsort(assign_cost[i])
        for f in cand:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                break
    # fallback: any feasible
    for i in range(n_cust):
        if result[i] < 0:
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    cost = _total_cost(result, demands, capacities, open_cost, assign_cost)
    if not np.isfinite(cost):
        return result, cost
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        best_gain = 1e-9
        best_i = -1
        best_f = -1
        for i in range(n_cust):
            cur = result[i]
            if time.time() > deadline:
                break
            # try moving i to each other facility
            for f in range(n_fac):
                if f == cur:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    # cost change:
                    # assignment: assign_cost[i,f] - assign_cost[i,cur]
                    d_assign = assign_cost[i, f] - assign_cost[i, cur]
                    # open cost change:
                    d_open = 0.0
                    # facility f newly opened?
                    if load[f] == 0:
                        d_open += open_cost[f]
                    # facility cur becomes empty?
                    if load[cur] == demands[i]:
                        d_open -= open_cost[cur]
                    total = d_assign + d_open
                    if total < -best_gain:
                        best_gain = -total
                        best_i = i
                        best_f = f
        if best_i >= 0:
            cur = result[best_i]
            load[cur] -= demands[best_i]
            load[best_f] += demands[best_i]
            result[best_i] = best_f
            cost -= best_gain
            improved = True
    return result, cost


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s

    best_result = None
    best_cost = np.inf

    modes = ['nearest', 'largest', 'effective', 'random', 'random']
    for mode in modes:
        if time.time() > deadline - 5.0:
            break
        res = _greedy(demands, capacities, open_cost, assign_cost, mode)
        res, c = _local_search(res, demands, capacities, open_cost, assign_cost, deadline)
        if np.isfinite(c) and c < best_cost:
            best_cost = c
            best_result = res.copy()

    if best_result is None:
        # last resort: largest-first greedy
        best_result = _greedy(demands, capacities, open_cost, assign_cost, 'largest')

    # final validation
    load = np.zeros(len(capacities))
    for i, f in enumerate(best_result):
        load[f] += demands[i]
    if np.any(load > capacities):
        # rebuild valid
        best_result = _greedy(demands, capacities, open_cost, assign_cost, 'largest')

    return best_result.astype(int)
# EVOLVE-BLOCK-END
