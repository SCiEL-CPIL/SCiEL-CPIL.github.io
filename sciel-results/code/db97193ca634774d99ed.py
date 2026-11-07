"""Capacitated facility location: soft-assignment (log-sum-exp) + capacity dual + local polish."""
import time
import numpy as np

DEADLINE_MARGIN = 10.0  # seconds to spare before the 150s wall-clock kill
T0 = time.time()


def _elapsed():
    return time.time() - T0


def _remaining():
    return 150.0 - DEADLINE_MARGIN - _elapsed()


def _is_valid(result, demands, capacities):
    if result is None:
        return False
    r = np.asarray(result)
    if r.shape != (len(demands),):
        return False
    if not np.issubdtype(r.dtype, np.integer):
        return False
    if (r < 0).any() or (r >= len(capacities)).any():
        return False
    load = np.zeros(len(capacities))
    for i, f in enumerate(r):
        load[f] += demands[i]
    return bool((load <= capacities).all())


def _cost_of(result, open_cost, assign_cost):
    r = np.asarray(result)
    total = float(assign_cost[np.arange(len(r)), r].sum())
    for f in set(r.tolist()):
        total += open_cost[f]
    return total


def _greedy_assign(demands, capacities, assign_cost):
    n = len(demands)
    result = np.zeros(n, dtype=int)
    load = np.zeros(len(capacities))
    for i in np.argsort(-demands):
        order = np.argsort(assign_cost[i])
        placed = False
        for f in order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # fallback: facility with most remaining capacity
            rem = capacities - load
            g = int(np.argmax(rem))
            load[g] += demands[i]
            result[i] = g
    return result


def _fix_capacity(result, demands, capacities, assign_cost):
    result = np.asarray(result, dtype=int).copy()
    load = np.zeros(len(capacities))
    for i, f in enumerate(result):
        load[f] += demands[i]
    for i in np.argsort(-demands):
        f = result[i]
        if load[f] > capacities[f]:
            order = np.argsort(assign_cost[i])
            for g in order:
                if g != f and load[g] + demands[i] <= capacities[g]:
                    load[f] -= demands[i]
                    load[g] += demands[i]
                    result[i] = g
                    break
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, cur_score):
    result = np.asarray(result, dtype=int).copy()
    load = np.zeros(len(capacities))
    for i, f in enumerate(result):
        load[f] += demands[i]
    improved = True
    while improved and _remaining() > 3.0:
        improved = False
        for i in np.arange(len(result)):
            f = result[i]
            order = np.argsort(assign_cost[i])
            for g in order:
                if g == f:
                    continue
                if load[g] + demands[i] <= capacities[g]:
                    new_load_f = load[f] - demands[i]
                    new_load_g = load[g] + demands[i]
                    delta_assign = assign_cost[i, g] - assign_cost[i, f]
                    delta_open = 0.0
                    if new_load_f == 0:
                        delta_open -= open_cost[f]
                    if load[g] == 0:
                        delta_open += open_cost[g]
                    if delta_assign + delta_open < -1e-9:
                        load[f] = new_load_f
                        load[g] = new_load_g
                        result[i] = g
                        cur_score += delta_assign + delta_open
                        improved = True
                        break
    return result, cur_score


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    n_customers = len(demands)
    n_facilities = len(capacities)

    best_result = None
    best_score = np.inf

    # Phase 0: greedy baseline
    g = _greedy_assign(demands, capacities, assign_cost)
    if _is_valid(g, demands, capacities):
        best_result = g.copy()
        best_score = _cost_of(best_result, open_cost, assign_cost)

    # Phase 1: soft assignment with capacity dual (Lagrangian / log-sum-exp surrogate)
    dual = np.zeros(n_facilities)
    for it in range(400):
        if _remaining() < 5.0:
            break
        eff = assign_cost + dual[np.newaxis, :]
        # log-sum-exp based softmax (smooth surrogate for argmin)
        temp = 1.0
        neg = -eff / temp
        neg_max = neg.max(axis=1, keepdims=True)
        p = np.exp(neg - neg_max)
        p = p / p.sum(axis=1, keepdims=True)
        load = p.sum(axis=0)
        step = 0.5
        dual = np.maximum(0.0, dual + step * (load - capacities))

        # hard assignment from eff, fix capacity
        res = np.argmin(eff, axis=1)
        res = _fix_capacity(res, demands, capacities, assign_cost)
        if _is_valid(res, demands, capacities):
            sc = _cost_of(res, open_cost, assign_cost)
            if sc < best_score:
                best_score = sc
                best_result = res.copy()

    # Phase 2: local polish
    if best_result is None:
        best_result = _greedy_assign(demands, capacities, assign_cost)
        best_score = _cost_of(best_result, open_cost, assign_cost)

    best_result, best_score = _local_search(
        best_result, demands, capacities, open_cost, assign_cost, best_score
    )

    if not _is_valid(best_result, demands, capacities):
        best_result = _greedy_assign(demands, capacities, assign_cost)

    return np.asarray(best_result, dtype=int)
