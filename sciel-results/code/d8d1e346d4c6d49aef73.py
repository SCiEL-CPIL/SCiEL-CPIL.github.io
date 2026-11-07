import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n_cust = len(demands)
    n_fac = len(capacities)

    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)

    # --- 1. LP Relaxation for Initialization ---
    # Variables: x[i, j] (n_cust * n_fac) and y[j] (n_fac)
    c_obj = np.concatenate([assign_cost.flatten(), open_cost])
    bounds = [(0, 1)] * (n_cust * n_fac + n_fac)

    A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
    for i in range(n_cust):
        for j in range(n_fac):
            A_eq[i, i * n_fac + j] = 1.0
    b_eq = np.ones(n_cust)

    n_cons_ub = n_fac + n_cust * n_fac
    A_ub = np.zeros((n_cons_ub, n_cust * n_fac + n_fac))
    b_ub = np.zeros(n_cons_ub)

    for j in range(n_fac):
        row_idx = j
        for i in range(n_cust):
            A_ub[row_idx, i * n_fac + j] = demands[i]
        A_ub[row_idx, n_cust * n_fac + j] = -capacities[j]

    for i in range(n_cust):
        for j in range(n_fac):
            row_idx = n_fac + i * n_fac + j
            A_ub[row_idx, i * n_fac + j] = 1.0
            A_ub[row_idx, n_cust * n_fac + j] = -1.0

    lp_solution = None
    try:
        if time.time() < deadline - 10:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.status == 0:
                lp_solution = res.x
    except Exception:
        lp_solution = None

    # --- 2. Initialization ---

    def compute_cost(result):
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(result):
            load[f] += demands[i]
            fac_count[f] += 1
        if np.any(load > capacities + 1e-9):
            return float('inf')
        return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), result])

    def greedy_init(order_func, randomize=False):
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(order_func(demands))

        for i in order:
            fac_order = np.argsort(assign_cost[i])
            if randomize:
                # Perturb costs slightly
                perturbed = assign_cost[i] + np.random.uniform(0, 1) * 10
                fac_order = np.argsort(perturbed)

            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break

            if not assigned:
                # Fallback
                remaining = capacities - load
                f = np.argmax(remaining)
                if remaining[f] >= demands[i]:
                    load[f] += demands[i]
                    result[i] = f
                else:
                    # Infeasible
                    return None

        return result

    def lp_rounding_init(lp_sol):
        if lp_sol is None:
            return None
        x_lp = lp_sol[:n_cust * n_fac].reshape(n_cust, n_fac)
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)

        # Sort by demand descending
        order = np.argsort(-demands)

        for i in order:
            # Get facilities sorted by LP value descending
            sorted_facs = np.argsort(-x_lp[i])
            assigned = False
            for f in sorted_facs:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break

            if not assigned:
                # Fallback to min cost feasible
                feasible = [f for f in range(n_fac) if load[f] + demands[i] <= capacities[f]]
                if feasible:
                    f_best = min(feasible, key=lambda f: assign_cost[i, f])
                    load[f_best] += demands[i]
                    result[i] = f_best
                else:
                    return None

        return result

    candidates = []

    # 1. LP Rounding
    init_lp = lp_rounding_init(lp_solution)
    if init_lp is not None:
        candidates.append(init_lp)

    # 2. Greedy Largest First
    init_g1 = greedy_init(lambda d: -d)
    if init_g1 is not None:
        candidates.append(init_g1)

    # 3. Greedy Smallest First
    init_g2 = greedy_init(lambda d: d)
    if init_g2 is not None:
        candidates.append(init_g2)

    # 4. Randomized Greedy (a few times)
    for _ in range(3):
        if time.time() > deadline - 20:
            break
        init_r = greedy_init(lambda d: -d, randomize=True)
        if init_r is not None:
            candidates.append(init_r)

    if not candidates:
        # Last resort: simple assignment
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    break
        candidates.append(result)

    # Pick best candidate
    best_init = min(candidates, key=compute_cost)

    # --- 3. Local Search Polish ---

    def _local_search(result, demands, capacities, open_cost, assign_cost, time_limit):
        n_cust = len(demands)
        n_fac = len(capacities)
        current = result.astype(int).copy()
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i, f in enumerate(current):
            load[f] += demands[i]
            fac_count[f] += 1

        def get_cost():
            return np.sum(open_cost[fac_count > 0]) + np.sum(assign_cost[np.arange(n_cust), current])

        current_cost = get_cost()
        best_cost = current_cost
        best_result = current.copy()

        start = time.time()
        improved = True

        while improved and (time.time() - start < time_limit):
            improved = False

            # 1. Single Reassignment
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0
                best_new_f = -1

                for new_f in range(n_fac):
                    if new_f == cur_f:
                        continue
                    if load[new_f] + d_i > capacities[new_f]:
                        continue

                    delta = assign_cost[i, new_f] - assign_cost[i, cur_f]

                    # Open cost change
                    if fac_count[cur_f] == 1:
                        delta -= open_cost[cur_f]
                    if fac_count[new_f] == 0:
                        delta += open_cost[new_f]

                    if delta < best_delta:
                        best_delta = delta
                        best_new_f = new_f

                if best_new_f != -1:
                    # Apply move
                    load[cur_f] -= d_i
                    load[best_new_f] += d_i
                    fac_count[cur_f] -= 1
                    fac_count[best_new_f] += 1
                    current[i] = best_new_f
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()

            # 2. Swap
            # To speed up, we can just iterate all pairs. n=100, pairs=4950.
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    return best_result
                f_i = current[i]
                d_i = demands[i]
                for j in range(i + 1, n_cust):
                    f_j = current[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]

                    # Check capacity after swap
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue

                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])

                    # Open cost change for swap between different facilities is 0
                    # (counts stay same)

                    if delta < -1e-9:
                        # Apply swap
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        current[i] = f_j
                        current[j] = f_i
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        break # Restart j loop for this i? Or just continue?
                        # Breaking j loop for this i is safer to avoid conflicting updates in same pass

        return best_result

    remaining_time = deadline - time.time()
    if remaining_time > 5:
        final_result = _local_search(best_init, demands, capacities, open_cost, assign_cost, remaining_time - 2)
    else:
        final_result = best_init

    # Final Validation
    load_final = np.zeros(n_fac)
    for i, f in enumerate(final_result):
        if f < 0 or f >= n_fac:
            return np.zeros(n_cust, dtype=int) # Invalid fallback
        load_final[f] += demands[i]

    if np.any(load_final > capacities + 1e-9):
        # If invalid, return the initial result which was valid
        return best_init

    return final_result
