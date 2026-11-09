"""Capacitated facility location: Multi-start local search with vectorized moves."""
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    deadline = time.time() + 140.0

    demands_arr = np.asarray(demands, dtype=np.float64)
    capacities_arr = np.asarray(capacities, dtype=np.float64)
    open_cost_arr = np.asarray(open_cost, dtype=np.float64)
    assign_cost_arr = np.asarray(assign_cost, dtype=np.float64)

    def calc_score(res):
        used = np.unique(res)
        return float(np.sum(open_cost_arr[used])) + float(np.sum(assign_cost_arr[np.arange(n_cust), res]))

    def loads_of(res):
        load = np.zeros(n_fac)
        np.add.at(load, res, demands_arr)
        return load

    def greedy_init(rng, noise_scale=0.0):
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        cust_order = np.argsort(-demands_arr)
        for i in cust_order:
            d_i = demands_arr[i]
            feasible = np.where(load + d_i <= capacities_arr)[0]
            if len(feasible) == 0:
                f = int(np.argmax(capacities_arr - load))
                res[i] = f
                load[f] += d_i
            else:
                costs = assign_cost_arr[i, feasible].copy()
                if noise_scale > 0:
                    costs += rng.random(len(costs)) * noise_scale
                f_idx = int(feasible[np.argmin(costs)])
                res[i] = f_idx
                load[f_idx] += d_i
        return res, load

    def single_moves_pass(res, load, time_limit):
        improved = False
        base_assign = assign_cost_arr[np.arange(n_cust), res]
        changed = True
        while changed:
            changed = False
            for i in range(n_cust):
                if time.time() > time_limit:
                    return res, load, improved
                f_old = res[i]
                d_i = demands_arr[i]
                feasible_mask = (load + d_i <= capacities_arr)
                feasible_mask = feasible_mask & (np.arange(n_fac) != f_old)
                if not np.any(feasible_mask):
                    continue
                delta_assign = assign_cost_arr[i, :] - base_assign[i]
                delta_open = np.zeros(n_fac)
                if load[f_old] == d_i:
                    delta_open[f_old] = -open_cost_arr[f_old]
                open_new = (load == 0) & (np.arange(n_fac) != f_old)
                if np.any(open_new):
                    delta_open[open_new] = open_cost_arr[open_new]
                total_delta = delta_assign + delta_open
                total_delta[~feasible_mask] = np.inf
                total_delta[f_old] = np.inf
                best_f_new = int(np.argmin(total_delta))
                best_delta = total_delta[best_f_new]
                if best_delta < -1e-9:
                    res[i] = best_f_new
                    load[f_old] -= d_i
                    load[best_f_new] += d_i
                    base_assign[i] = assign_cost_arr[i, best_f_new]
                    changed = True
                    improved = True
        return res, load, improved

    def close_facility_pass(res, load, time_limit):
        improved = False
        for f_close in range(n_fac):
            if time.time() > time_limit:
                break
            if load[f_close] == 0:
                continue
            custs = np.where(res == f_close)[0]
            if len(custs) == 0:
                continue

            temp_load = load.copy()
            temp_load[f_close] = 0
            temp_res = res.copy()

            cust_order = custs[np.argsort(-demands_arr[custs])]
            success = True
            for c in cust_order:
                d_c = demands_arr[c]
                feasible = np.where(temp_load + d_c <= capacities_arr)[0]
                if len(feasible) == 0:
                    success = False
                    break
                costs = assign_cost_arr[c, feasible]
                open_penalty = np.where(temp_load[feasible] == 0, open_cost_arr[feasible], 0.0)
                adjusted_costs = costs + open_penalty
                best_f = feasible[np.argmin(adjusted_costs)]
                temp_res[c] = best_f
                temp_load[best_f] += d_c

            if not success:
                continue
            if np.any(temp_load > capacities_arr):
                continue

            old_score = calc_score(res)
            new_score = calc_score(temp_res)

            if new_score < old_score - 1e-9:
                res = temp_res
                load = temp_load
                improved = True
        return res, load, improved

    def local_search(res, load, time_limit):
        improved = True
        while improved:
            improved = False
            if time.time() > time_limit:
                break
            res, load, imp1 = single_moves_pass(res, load, time_limit)
            improved = improved or imp1
            if time.time() > time_limit:
                break
            res, load, imp2 = close_facility_pass(res, load, time_limit)
            improved = improved or imp2
        return res, load

    # Multi-start
    rng = np.random.default_rng(42)
    best_result = None
    best_score = np.inf

    # Start 1: Deterministic greedy
    res, load = greedy_init(rng)
    res, load = local_search(res.copy(), load.copy(), deadline - 5.0)
    if np.all(load <= capacities_arr):
        s = calc_score(res)
        if s < best_score:
            best_score = s
            best_result = res.copy()

    # Additional starts with noise
    noise_levels = [5.0, 10.0, 15.0, 20.0]
    for noise in noise_levels:
        if time.time() > deadline - 20.0:
            break
        res, load = greedy_init(rng, noise_scale=noise)
        if np.all(load <= capacities_arr):
            res, load = local_search(res.copy(), load.copy(), deadline - 5.0)
            if np.all(load <= capacities_arr):
                s = calc_score(res)
                if s < best_score:
                    best_score = s
                    best_result = res.copy()

    # Random restarts
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
        res_init = np.zeros(n_cust, dtype=int)
        load_init = np.zeros(n_fac)
        cust_order = np.random.permutation(n_cust)
        for i in cust_order:
            d_i = demands_arr[i]
            feasible = np.where(load_init + d_i <= capacities_arr)[0]
            if len(feasible) == 0:
                f = int(np.argmax(capacities_arr - load_init))
                res_init[i] = f
                load_init[f] += d_i
            else:
                f_idx = int(feasible[rng.integers(0, len(feasible))])
                res_init[i] = f_idx
                load_init[f_idx] += d_i

        if np.all(load_init <= capacities_arr):
            res, load = local_search(res_init.copy(), load_init.copy(), deadline - 5.0)
            if np.all(load <= capacities_arr):
                s = calc_score(res)
                if s < best_score:
                    best_score = s
                    best_result = res.copy()

    # Final polish
    if best_result is not None and time.time() < deadline - 3.0:
        res_final, load_final = local_search(best_result.copy(), loads_of(best_result), deadline - 2.0)
        if np.all(load_final <= capacities_arr):
            s = calc_score(res_final)
            if s < best_score:
                best_result = res_final
                best_score = s

    return best_result
