"""Capacitated facility location: Multi-start ILS with vectorized moves + close-facility."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using iterated local search with
    vectorized single moves, swap moves, pair moves, and close-facility moves.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    rng = np.random.default_rng(12345)

    # Safety margin: 150s limit, aim for ~140s max to leave buffer
    deadline = time.time() + 140.0

    def calc_cost(res):
        used = np.unique(res)
        return float(np.sum(open_cost[used])) + float(np.sum(assign_cost[np.arange(n_cust), res]))

    def loads_of(res):
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        return load

    # ---- Greedy seed (always valid fallback) ----
    result_greedy = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    order = np.argsort(-demands)
    for i in order:
        feasible = np.where(load + demands[i] <= capacities)[0]
        if len(feasible) == 0:
            f = int(np.argmax(capacities - load))
            result_greedy[i] = f
            load[f] += demands[i]
        else:
            costs = assign_cost[i, feasible]
            f_idx = int(feasible[np.argmin(costs)])
            result_greedy[i] = f_idx
            load[f_idx] += demands[i]

    best_result = result_greedy.copy()
    best_score = calc_cost(best_result)

    # ---- Vectorized single-move best-improvement pass ----
    def single_moves_pass(res, load):
        improved = False
        # Repeated best-improvement until no improving move in a full sweep
        while True:
            if time.time() > deadline - 2.0:
                break
            any_improve = False
            base_assign = assign_cost[np.arange(n_cust), res]
            for i in range(n_cust):
                if time.time() > deadline - 2.0:
                    break
                f_old = res[i]
                d_i = demands[i]
                feasible_mask = (load + d_i <= capacities)
                feasible_mask = feasible_mask & (np.arange(n_fac) != f_old)
                if not np.any(feasible_mask):
                    continue
                delta_assign = assign_cost[i, :] - base_assign[i]
                delta_open = np.zeros(n_fac)
                if load[f_old] == d_i:
                    delta_open[f_old] = -open_cost[f_old]
                open_new = (load == 0) & (np.arange(n_fac) != f_old)
                if np.any(open_new):
                    delta_open[open_new] = open_cost[open_new]
                total_delta = delta_assign + delta_open
                total_delta[~feasible_mask] = np.inf
                total_delta[f_old] = np.inf
                best_f_new = int(np.argmin(total_delta))
                best_delta = total_delta[best_f_new]
                if best_delta < -1e-9:
                    res[i] = best_f_new
                    load[f_old] -= d_i
                    load[best_f_new] += d_i
                    any_improve = True
                    improved = True
            if not any_improve:
                break
        return res, load, improved

    # ---- Swap moves (sampled) ----
    def swap_pass(res, load, n_samples=1500):
        improved = False
        for _ in range(n_samples):
            if time.time() > deadline - 2.0:
                break
            i, j = rng.integers(0, n_cust, size=2)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            f_i = res[i]
            f_j = res[j]
            if f_i == f_j:
                continue
            d_i = demands[i]
            d_j = demands[j]
            if load[f_j] - d_j + d_i > capacities[f_j]:
                continue
            if load[f_i] - d_i + d_j > capacities[f_i]:
                continue
            delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                           (assign_cost[j, f_i] - assign_cost[j, f_j])
            # open cost: only changes if a facility becomes empty after losing its sole customer
            # f_i loses i (gain j): empty only if load[f_i]==d_i and d_j==0 (impossible)
            # f_j loses j (gain i): empty only if load[f_j]==d_j and d_i==0 (impossible)
            total_delta = delta_assign
            if total_delta < -1e-9:
                res[i], res[j] = res[j], res[i]
                load[f_i] -= d_i
                load[f_i] += d_j
                load[f_j] -= d_j
                load[f_j] += d_i
                improved = True
        return res, load, improved

    # ---- Pair move: move two customers to two (possibly same) other facilities ----
    def pair_move_pass(res, load, n_samples=800):
        improved = False
        for _ in range(n_samples):
            if time.time() > deadline - 2.0:
                break
            i, j = rng.integers(0, n_cust, size=2)
            if i == j:
                continue
            f_i = res[i]
            f_j = res[j]
            d_i = demands[i]
            d_j = demands[j]
            # Candidate target facilities for each (must have capacity after both moves)
            # Try a small set: for i, best 2 feasible; for j, best 2 feasible
            feas_i = np.where(load + d_i <= capacities)[0]
            feas_i = feas_i[feas_i != f_i]
            feas_j = np.where(load - d_j + d_j <= capacities)[0]  # placeholder; recompute below
            # Actually feasibility depends on joint move; we'll check explicitly.
            # Enumerate a limited set of (gi, gj)
            if len(feas_i) == 0:
                continue
            # pick up to 3 cheapest for i
            top_i = feas_i[np.argsort(assign_cost[i, feas_i])][:3]
            # for j, feasible ignoring i's move first
            feas_j0 = np.where(load + d_j <= capacities)[0]
            feas_j0 = feas_j0[feas_j0 != f_j]
            if len(feas_j0) == 0:
                continue
            top_j = feas_j0[np.argsort(assign_cost[j, feas_j0])][:3]
            best_delta = 0.0
            best_gi = -1
            best_gj = -1
            for gi in top_i:
                # after moving i: load' = load - d_i at f_i, + d_i at gi
                # capacity for j: load'[gj] + d_j <= capacities[gj]
                # load'[gj] = load[gj] - (d_j if gj==f_j else 0) + (d_i if gj==gi else 0)
                for gj in top_j:
                    if gj == f_j and gj == f_i:
                        # j stays? no, gj must != f_j
                        pass
                    if gj == f_j:
                        continue
                    # compute new load at gj
                    load_gj = load[gj]
                    if gj == f_j:
                        load_gj -= d_j
                    if gj == gi:
                        load_gj += d_i
                    if load_gj + d_j > capacities[gj]:
                        continue
                    # also gi capacity already checked via feas_i
                    delta = (assign_cost[i, gi] - assign_cost[i, f_i]) + \
                            (assign_cost[j, gj] - assign_cost[j, f_j])
                    # open cost changes
                    # f_i: loses i, maybe gains j if gj==f_i
                    # f_j: loses j, maybe gains i if gi==f_j
                    # compute open delta by tracking which facilities become empty/open
                    # Simpler: compute open cost before and after for affected facilities
                    affected = set()
                    affected.add(f_i); affected.add(f_j); affected.add(gi); affected.add(gj)
                    # simulate loads
                    old_load = load.copy()
                    new_load = load.copy()
                    new_load[f_i] -= d_i
                    new_load[gi] += d_i
                    new_load[f_j] -= d_j
                    new_load[gj] += d_j
                    if np.any(new_load < 0) or np.any(new_load > capacities):
                        continue
                    old_open = sum(open_cost[f] for f in affected if old_load[f] > 0)
                    new_open = sum(open_cost[f] for f in affected if new_load[f] > 0)
                    delta += (new_open - old_open)
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_gi = gi
                        best_gj = gj
            if best_gi >= 0 and best_gj >= 0 and best_delta < -1e-9:
                # apply
                load[f_i] -= d_i
                load[best_gi] += d_i
                load[f_j] -= d_j
                load[best_gj] += d_j
                res[i] = best_gi
                res[j] = best_gj
                improved = True
        return res, load, improved

    # ---- Close-facility move (vectorized reassignment with open-penalty) ----
    def close_facility_pass(res, load):
        improved = False
        for f_close in range(n_fac):
            if time.time() > deadline - 2.0:
                break
            if load[f_close] == 0:
                continue
            custs = np.where(res == f_close)[0]
            if len(custs) == 0:
                continue
            temp_load = load.copy()
            temp_load[f_close] = 0
            temp_res = res.copy()
            cust_order = custs[np.argsort(-demands[custs])]
            success = True
            for c in cust_order:
                d_c = demands[c]
                feasible = np.where(temp_load + d_c <= capacities)[0]
                if len(feasible) == 0:
                    success = False
                    break
                costs = assign_cost[c, feasible]
                open_penalty = np.where(temp_load[feasible] == 0, open_cost[feasible], 0.0)
                adjusted = costs + open_penalty
                best_f = int(feasible[np.argmin(adjusted)])
                temp_res[c] = best_f
                temp_load[best_f] += d_c
            if not success:
                continue
            if np.any(temp_load > capacities):
                continue
            old_assign_sum = float(np.sum(assign_cost[custs, f_close]))
            new_assign_sum = float(np.sum(assign_cost[custs, temp_res[custs]]))
            # open cost delta
            old_open = float(np.sum(open_cost[load > 0]))
            new_open = float(np.sum(open_cost[temp_load > 0]))
            delta = (new_assign_sum - old_assign_sum) + (new_open - old_open)
            if delta < -1e-9:
                res[:] = temp_res
                load[:] = temp_load
                improved = True
        return res, load, improved

    # ---- Full local search (iterate until no improvement) ----
    def local_search(res, load):
        res = res.copy()
        load = load.copy()
        improved = True
        stall = 0
        while improved:
            improved = False
            if time.time() > deadline - 2.0:
                break
            # Single moves (best-improvement)
            res, load, imp = single_moves_pass(res, load)
            improved = improved or imp
            # Swap moves
            res, load, imp = swap_pass(res, load, n_samples=1500)
            improved = improved or imp
            # Pair moves
            res, load, imp = pair_move_pass(res, load, n_samples=800)
            improved = improved or imp
            # Close facility
            res, load, imp = close_facility_pass(res, load)
            improved = improved or imp
            if not improved:
                break
        return res, load

    # ---- Multi-start: different alpha penalties ----
    alphas = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break
        eff_cost = assign_cost + alpha * open_cost
        init_res = np.argmin(eff_cost, axis=1).astype(int)
        res = init_res.copy()
        load = loads_of(res)
        # Repair capacity violations
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            overloaded = np.where(load > capacities)[0]
            if len(overloaded) == 0:
                break
            f = overloaded[0]
            custs_in_f = np.where(res == f)[0]
            if len(custs_in_f) == 0:
                break
            best_cust = -1
            best_g = -1
            min_penalty = np.inf
            for i in custs_in_f:
                for g in range(n_fac):
                    if g == f:
                        continue
                    if load[g] + demands[i] <= capacities[g]:
                        penalty = assign_cost[i, g] - assign_cost[i, f]
                        if penalty < min_penalty:
                            min_penalty = penalty
                            best_cust = i
                            best_g = g
            if best_cust == -1:
                break
            res[best_cust] = best_g
            load[f] -= demands[best_cust]
            load[best_g] += demands[best_cust]
        if np.all(load <= capacities):
            res, load = local_search(res, load)
            score = calc_cost(res)
            if score < best_score:
                best_score = score
                best_result = res.copy()

    # ---- Final polish: a few more local search passes on the best ----
    current_res = best_result.copy()
    current_load = loads_of(current_res)
    for _ in range(4):
        if time.time() > deadline - 5.0:
            break
        current_res, current_load = local_search(current_res, current_load)
        score = calc_cost(current_res)
        if score < best_score:
            best_score = score
            best_result = current_res.copy()
        else:
            break

    # Validate and return
    final_load = loads_of(best_result)
    if np.any(final_load > capacities):
        # Fallback to greedy which is valid
        return result_greedy
    return best_result.astype(int)
# EVOLVE-BLOCK-END
