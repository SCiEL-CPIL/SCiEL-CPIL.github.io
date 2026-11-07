"""Capacitated facility location: assign 100 customers to 20 facilities."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_init(demands, capacities, open_cost, assign_cost, rng):
    """Greedy: largest demand first, cheapest feasible facility (random tie-break)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=np.float64)
    order = np.argsort(-demands, kind='stable')
    for i in order:
        d = demands[i]
        costs = assign_cost[i]
        # feasible facilities sorted by (cost, random)
        feasible = [f for f in range(n_fac) if load[f] + d <= capacities[f]]
        if not feasible:
            # fallback: any facility with most remaining capacity
            f = int(np.argmax(capacities - load))
            load[f] += d
            result[i] = f
            continue
        # sort by cost with random tie-break
        fs = np.array(feasible)
        key = costs[fs] + rng.uniform(0, 1e-6, size=fs.shape)
        best = fs[np.argmin(key)]
        load[best] += d
        result[i] = best
    return result, load

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    n_cust = len(result)
    total = float(np.sum(assign_cost[np.arange(n_cust), result]))
    used = np.unique(result)
    total += float(np.sum(open_cost[used]))
    return total

def _is_valid(result, demands, capacities):
    n_fac = len(capacities)
    if np.any(result < 0) or np.any(result >= n_fac):
        return False
    load = np.zeros(n_fac, dtype=np.float64)
    for i, f in enumerate(result):
        load[f] += demands[i]
    return bool(np.all(load <= capacities + 1e-9))

def _local_search(result, load, demands, capacities, open_cost, assign_cost, deadline):
    """First-improvement single-move + swap local search. Mutates result/load in place."""
    n_cust = len(demands)
    n_fac = len(capacities)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in np.random.permutation(n_cust):
            if time.time() > deadline:
                break
            cf = result[i]
            di = demands[i]
            # --- single move ---
            moved = False
            for f in range(n_fac):
                if f == cf:
                    continue
                if load[f] + di > capacities[f] + 1e-9:
                    continue
                assign_delta = assign_cost[i, f] - assign_cost[i, cf]
                open_delta = 0.0
                if load[cf] == di:          # cf becomes empty
                    open_delta -= open_cost[cf]
                if load[f] == 0:            # f was empty
                    open_delta += open_cost[f]
                if assign_delta + open_delta < -1e-9:
                    load[cf] -= di
                    load[f] += di
                    result[i] = f
                    improved = True
                    moved = True
                    break
            if moved:
                continue
            # --- swap: find j on cf (j != i) and f such that swap improves ---
            # candidates j sharing facility cf
            same = [j for j in range(n_cust) if j != i and result[j] == cf]
            if not same:
                continue
            for j in same:
                dj = demands[j]
                # after swapping i->f and j->cf:
                # cf load: -di + dj ; f load: +di - dj
                new_load_cf = load[cf] - di + dj
                new_load_f = load[f] + di - dj
                if new_load_cf > capacities[cf] + 1e-9:
                    continue
                if new_load_f > capacities[f] + 1e-9:
                    continue
                if new_load_cf < -1e-9 or new_load_f < -1e-9:
                    continue
                assign_delta = (assign_cost[i, f] + assign_cost[j, cf]) - (assign_cost[i, cf] + assign_cost[j, f])
                open_delta = 0.0
                if load[cf] == di:          # cf becomes empty (only i was there)
                    open_delta -= open_cost[cf]
                if load[f] == 0:            # f becomes empty (only j was there)
                    open_delta -= open_cost[f]
                # no new facility opened by a swap (both cf and f already used)
                if assign_delta + open_delta < -1e-9:
                    load[cf] = new_load_cf
                    load[f] = new_load_f
                    result[i] = f
                    result[j] = cf
                    improved = True
                    break
            # continue to next i
    return result, load

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    n_cust = len(demands)
    n_fac = len(capacities)

    deadline = time.time() + 135.0   # safety margin before 150s kill
    best_result = None
    best_cost = np.inf
    rng = np.random.default_rng(12345)

    for attempt in range(40):
        if time.time() > deadline:
            break
        result, load = _greedy_init(demands, capacities, open_cost, assign_cost, rng)
        if not _is_valid(result, demands, capacities):
            continue
        # local search with a sub-deadline so we always finish the attempt
        sub_deadline = min(deadline, time.time() + 20.0)
        result, load = _local_search(result, load, demands, capacities, open_cost, assign_cost, sub_deadline)
        if not _is_valid(result, demands, capacities):
            continue
        c = _compute_cost(result, demands, capacities, open_cost, assign_cost)
        if c < best_cost:
            best_cost = c
            best_result = result.copy()
        # early stop if we've used too much time
        if time.time() > deadline - 2.0:
            break

    if best_result is None:
        # deterministic fallback greedy
        best_result, _ = _greedy_init(demands, capacities, open_cost, assign_cost, np.random.default_rng(0))

    return best_result
# EVOLVE-BLOCK-END
