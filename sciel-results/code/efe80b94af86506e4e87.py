"""Capacitated facility location: MILP exact solve + fast local search polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_result = None
    best_cost = float('inf')

    def _total_cost(result):
        load = np.zeros(n_fac)
        for c in range(n_cust):
            load[result[c]] += demands[c]
        used = load > 0
        open_sum = open_cost[used].sum()
        assign_sum = np.sum(assign_cost[np.arange(n_cust), result])
        return open_sum + assign_sum

    def _greedy_seed(strategy='eff'):
        """Generate initial assignment using various greedy strategies."""
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac, dtype=float)

        amortized = open_cost / np.maximum(capacities, 1.0)

        if strategy == 'eff':
            eff_cost = assign_cost + np.outer(demands, amortized)
            order = np.argsort(-demands)
            cost_matrix = eff_cost
        elif strategy == 'raw':
            order = np.argsort(-demands)
            cost_matrix = assign_cost
        elif strategy == 'random':
            order = np.random.permutation(n_cust)
            cost_matrix = assign_cost
        elif strategy == 'small_first':
            order = np.argsort(demands)
            cost_matrix = assign_cost
        elif strategy == 'cheapest':
            min_costs = np.min(assign_cost, axis=1)
            order = np.argsort(min_costs)
            cost_matrix = assign_cost
        else:
            order = np.argsort(-demands)
            cost_matrix = assign_cost

        for i in order:
            fac_order = np.argsort(cost_matrix[i])
            assigned = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    result[i] = f
                    assigned = True
                    break
            if not assigned:
                for f in np.argsort(assign_cost[i]):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        result[i] = f
                        break
            if result[i] < 0:
                remaining = capacities - load
                f = int(np.argmax(remaining))
                result[i] = f
                load[f] += demands[i]

        return result

    def _milp_solve(time_budget=100.0):
        """Solve the capacitated facility location problem exactly via MILP."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
        except ImportError:
            return None

        milp_start = time.time()
        milp_deadline = milp_start + time_budget

        # Variables: x[c*n_fac + f] for assignment, y[f] for facility open
        n_x = n_cust * n_fac
        n_y = n_fac
        n_vars = n_x + n_y

        # Objective
        c_obj = np.zeros(n_vars)
        c_obj[:n_x] = assign_cost.flatten()
        c_obj[n_x:] = open_cost

        # Build constraint matrix efficiently using sparse-friendly approach
        # We'll build rows for: assignment (=1), capacity (<=0), linking (<=0)
        n_constraints = n_cust + n_fac + n_cust * n_fac
        A = np.zeros((n_constraints, n_vars))
        lb = np.full(n_constraints, -np.inf)
        ub = np.full(n_constraints, np.inf)

        # Assignment constraints: sum_f x[c,f] = 1
        for c in range(n_cust):
            row_idx = c
            for f in range(n_fac):
                A[row_idx, c * n_fac + f] = 1.0
            lb[row_idx] = 1.0
            ub[row_idx] = 1.0

        # Capacity constraints: sum_c demands[c]*x[c,f] - capacities[f]*y[f] <= 0
        for f in range(n_fac):
            row_idx = n_cust + f
            for c in range(n_cust):
                A[row_idx, c * n_fac + f] = demands[c]
            A[row_idx, n_x + f] = -capacities[f]
            ub[row_idx] = 0.0

        # Linking constraints: x[c,f] - y[f] <= 0
        for c in range(n_cust):
            for f in range(n_fac):
                row_idx = n_cust + n_fac + c * n_fac + f
                A[row_idx, c * n_fac + f] = 1.0
                A[row_idx, n_x + f] = -1.0
                ub[row_idx] = 0.0

        integrality = np.ones(n_vars)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        constraint = LinearConstraint(A, lb, ub)

        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds)

        if time.time() > deadline - 5.0:
            return None

        if res.status == 0 and res.x is not None:
            new_result = np.zeros(n_cust, dtype=int)
            for c in range(n_cust):
                vals = res.x[c * n_fac:(c + 1) * n_fac]
                new_result[c] = int(np.argmax(vals))

            # Validate feasibility
            load = np.zeros(n_fac)
            for c in range(n_cust):
                load[new_result[c]] += demands[c]
            if np.all(load <= capacities + 1e-6):
                return new_result

        return None

    def _local_search(result, deadline):
        """Fast best-improvement local search with Relocate and Swap moves."""
        result = result.copy()
        load = np.zeros(n_fac)
        count = np.zeros(n_fac, dtype=int)
        for c in range(n_cust):
            load[result[c]] += demands[c]
            count[result[c]] += 1

        current_cost = _total_cost(result)

        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False

            # --- Phase 1: Best Relocate ---
            best_delta = 0.0
            best_c = -1
            best_f = -1

            for c in range(n_cust):
                if time.time() > deadline - 5.0:
                    break
                old_f = result[c]
                d_c = demands[c]
                base_cost = assign_cost[c, old_f]

                for f in range(n_fac):
                    if f == old_f:
                        continue
                    if load[f] + d_c > capacities[f]:
                        continue

                    delta = assign_cost[c, f] - base_cost

                    # Open cost changes
                    if count[old_f] == 1:
                        delta -= open_cost[old_f]
                    if count[f] == 0:
                        delta += open_cost[f]

                    if delta < best_delta:
                        best_delta = delta
                        best_c = c
                        best_f = f

            if best_delta < -1e-9 and best_c != -1:
                c = best_c
                f = best_f
                old_f = result[c]
                load[old_f] -= demands[c]
                count[old_f] -= 1
                load[f] += demands[c]
                count[f] += 1
                result[c] = f
                current_cost += best_delta
                improved = True
                continue

            # --- Phase 2: Best Swap ---
            best_swap_delta = 0.0
            best_c1 = -1
            best_c2 = -1

            for c1 in range(n_cust):
                if time.time() > deadline - 5.0:
                    break
                f1 = result[c1]
                d1 = demands[c1]
                ac1_f1 = assign_cost[c1, f1]

                for c2 in range(c1 + 1, n_cust):
                    f2 = result[c2]
                    if f1 == f2:
                        continue
                    d2 = demands[c2]

                    new_load_f1 = load[f1] - d1 + d2
                    new_load_f2 = load[f2] - d2 + d1

                    if new_load_f1 > capacities[f1] or new_load_f2 > capacities[f2]:
                        continue

                    delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                            (ac1_f1 + assign_cost[c2, f2])

                    if delta < best_swap_delta:
                        best_swap_delta = delta
                        best_c1 = c1
                        best_c2 = c2

            if best_swap_delta < -1e-9 and best_c1 != -1:
                c1 = best_c1
                c2 = best_c2
                f1 = result[c1]
                f2 = result[c2]

                load[f1] = load[f1] - demands[c1] + demands[c2]
                load[f2] = load[f2] - demands[c2] + demands[c1]
                result[c1] = f2
                result[c2] = f1
                current_cost += best_swap_delta
                improved = True
                continue

        return result, current_cost

    # Step 1: Try MILP exact solve (primary structural change)
    if time.time() < deadline - 20.0:
        milp_result = _milp_solve(time_budget=100.0)
        if milp_result is not None:
            milp_cost = _total_cost(milp_result)
            if milp_cost < best_cost:
                best_cost = milp_cost
                best_result = milp_result.copy()

            # Polish MILP result with local search
            if time.time() < deadline - 10.0:
                ls_result, ls_cost = _local_search(milp_result, deadline)
                if ls_cost < best_cost:
                    best_cost = ls_cost
                    best_result = ls_result

    # Step 2: If MILP didn't give a great answer (or failed), use greedy seeds + local search
    if time.time() < deadline - 5.0:
        strategies = ['eff', 'raw', 'cheapest']
        for strat in strategies:
            if time.time() >= deadline - 5.0:
                break

            seed_result = _greedy_seed(strat)
            if np.any(seed_result < 0):
                continue

            seed_cost = _total_cost(seed_result)
            if seed_cost < best_cost:
                best_cost = seed_cost
                best_result = seed_result.copy()

            # Local search on this seed
            if time.time() < deadline - 8.0:
                ls_result, ls_cost = _local_search(seed_result, deadline)
                if ls_cost < best_cost:
                    best_cost = ls_cost
                    best_result = ls_result

    # Final fallback
    if best_result is None or np.any(best_result < 0):
        best_result = _greedy_seed('eff')

    # Final validation
    if np.any(best_result < 0) or np.any(best_result >= n_fac):
        best_result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        for i in range(n_cust):
            for f in range(n_fac):
                if load[f] + demands[i] <= capacities[f]:
                    best_result[i] = f
                    load[f] += demands[i]
                    break
            if best_result[i] == 0 and load[0] + demands[i] > capacities[0]:
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        best_result[i] = f
                        load[f] += demands[i]
                        break

    return best_result.astype(int)
# EVOLVE-BLOCK-END
