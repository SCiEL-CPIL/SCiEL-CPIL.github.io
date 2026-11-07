"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _objective(result, demands, capacities, open_cost, assign_cost):
    """Compute objective; return inf if invalid."""
    n_c = len(demands)
    n_f = len(capacities)
    if len(result) != n_c:
        return float('inf')
    load = np.zeros(n_f)
    for i in range(n_c):
        f = int(result[i])
        if f < 0 or f >= n_f:
            return float('inf')
        load[f] += demands[i]
    if np.any(load > capacities + 1e-9):
        return float('inf')
    used = np.zeros(n_f, dtype=bool)
    for f in result:
        used[f] = True
    return float(np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_c), result]))


def _greedy(demands, capacities, assign_cost, order):
    """Greedy largest-first assignment following given customer order."""
    n_f = len(capacities)
    result = np.zeros(len(demands), dtype=int)
    load = np.zeros(n_f)
    remaining = np.full(n_f, capacities, dtype=float)
    for i in order:
        d = demands[i]
        # feasible facilities: remaining capacity >= d
        feas = remaining >= d
        if not np.any(feas):
            # fallback: pick facility with max remaining
            f = int(np.argmax(remaining))
        else:
            # among feasible, cheapest assign cost
            costs = assign_cost[i].copy()
            costs[~feas] = np.inf
            f = int(np.argmin(costs))
        result[i] = f
        remaining[f] -= d
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """In-place local search: relocate + swap passes. Returns objective."""
    n_c = len(demands)
    n_f = len(capacities)
    result = result.copy()
    load = np.zeros(n_f, dtype=float)
    for i in range(n_c):
        load[int(result[i])] += demands[i]
    used = np.zeros(n_f, dtype=bool)
    for f in result:
        used[f] = True
    obj = float(np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_c), result]))

    improved = True
    while improved:
        improved = False
        # ---- Relocate passes ----
        for i in range(n_c):
            if time.time() > deadline:
                break
            fi = int(result[i])
            di = demands[i]
            cur = assign_cost[i, fi]
            # best alternative facility
            alt = np.full(n_f, np.inf)
            for f in range(n_f):
                if f == fi:
                    continue
                if load[f] + di <= capacities[f] + 1e-9:
                    alt[f] = assign_cost[i, f]
            if np.isinf(np.min(alt)):
                continue
            best_f = int(np.argmin(alt))
            delta_assign = alt[best_f] - cur
            # opening cost change
            if load[fi] == di:  # facility fi becomes empty
                delta_open = -open_cost[fi]
            elif not used[best_f]:  # opening a new facility
                delta_open = open_cost[best_f]
            else:
                delta_open = 0.0
            if delta_assign + delta_open < -1e-9:
                # apply
                load[fi] -= di
                load[best_f] += di
                if load[fi] == 0:
                    used[fi] = False
                used[best_f] = True
                result[i] = best_f
                obj += delta_assign + delta_open
                improved = True
        if time.time() > deadline:
            break
        # ---- Swap passes ----
        for i in range(n_c):
            if time.time() > deadline:
                break
            fi = int(result[i])
            di = demands[i]
            for j in range(i + 1, n_c):
                if time.time() > deadline:
                    break
                fj = int(result[j])
                if fj == fi:
                    continue
                dj = demands[j]
                # capacity feasibility after swap
                if load[fi] - di + dj > capacities[fi] + 1e-9:
                    continue
                if load[fj] - dj + di > capacities[fj] + 1e-9:
                    continue
                cur = assign_cost[i, fi] + assign_cost[j, fj]
                new = assign_cost[i, fj] + assign_cost[j, fi]
                delta_assign = new - cur
                # opening cost change
                delta_open = 0.0
                if load[fi] == di and not (load[fj] - dj + di == 0):
                    # fi empties
                    pass
                # recompute used after hypothetical swap
                new_load_fi = load[fi] - di + dj
                new_load_fj = load[fj] - dj + di
                old_used_fi = used[fi]
                old_used_fj = used[fj]
                new_used_fi = new_load_fi > 0
                new_used_fj = new_load_fj > 0
                if old_used_fi and not new_used_fi:
                    delta_open -= open_cost[fi]
                if (not old_used_fi) and new_used_fi:
                    delta_open += open_cost[fi]
                if old_used_fj and not new_used_fj:
                    delta_open -= open_cost[fj]
                if (not old_used_fj) and new_used_fj:
                    delta_open += open_cost[fj]
                if delta_assign + delta_open < -1e-9:
                    # apply swap
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    used[fi] = new_used_fi
                    used[fj] = new_used_fj
                    result[i] = fj
                    result[j] = fi
                    obj += delta_assign + delta_open
                    improved = True
    return obj, result


def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s
    n_c = len(demands)
    n_f = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    best_obj = float('inf')
    best_result = None

    # Base greedy (largest demand first)
    order = np.argsort(-demands)
    res = _greedy(demands, capacities, assign_cost, order)
    obj, res = _local_search(res, demands, capacities, open_cost, assign_cost, deadline)
    if obj < best_obj:
        best_obj = obj
        best_result = res.copy()

    # Additional restarts with randomized perturbations while time remains
    rng = np.random.default_rng(12345)
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        # perturb: random order biased toward large demand
        noise = rng.random(n_c) * 0.3
        weights = -demands + noise * (demands.max() - demands.min() + 1)
        order = np.argsort(-weights)
        res = _greedy(demands, capacities, assign_cost, order)
        obj, res = _local_search(res, demands, capacities, open_cost, assign_cost, deadline)
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_result = res.copy()

    if best_result is None:
        # final safety: pure greedy
        best_result = _greedy(demands, capacities, assign_cost, np.argsort(-demands))

    return best_result
# EVOLVE-BLOCK-END
