"""
Facility location solver: MIP + LP relaxation + local search with 2-swaps.
Strategy:
  1. Try MIP (scipy.optimize.milp) for exact solution.
  2. If MIP fails/times out, solve LP relaxation and round.
  3. Apply local search (single moves + 2-swaps) to best solution found.
  4. Greedy fallback if all else fails.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def compute_cost(assignment):
        """Compute total cost, return inf if infeasible."""
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(assignment[c])
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                return np.inf
        cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        for f in range(n_facilities):
            if used[f]:
                cost += open_cost[f]
        return cost

    def greedy_assignment(order=None):
        """Greedy: assign in given order to cheapest feasible facility."""
        if order is None:
            order = np.arange(n_customers)
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            f = int(np.argmin(costs))
            if feasible[f]:
                result[c] = f
                load[f] += demands[c]
            else:
                return None
        return result

    def local_search(assignment, max_iter=500):
        """Improve via single reassignments and 2-swap moves."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment

        for it in range(max_iter):
            if time.time() > deadline - 8:
                break
            improved = False

            # --- Single-move phase ---
            for c in range(n_customers):
                old_f = int(assignment[c])
                load[old_f] -= demands[c]
                best_new_f = old_f
                best_delta = 0.0
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                if best_new_f != old_f and best_delta < -1e-9:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]

            # --- 2-swap phase ---
            if time.time() > deadline - 12:
                break
            n_swaps_to_try = min(3000, n_customers * (n_customers - 1) // 2)
            rng = np.random.RandomState(it * 7919 + 13)
            tried = set()
            swap_improved = False
            for _ in range(n_swaps_to_try):
                if time.time() > deadline - 8:
                    break
                c1 = rng.randint(n_customers)
                c2 = rng.randint(n_customers)
                if c1 == c2 or (c1, c2) in tried:
                    continue
                tried.add((c1, c2))
                f1 = int(assignment[c1])
                f2 = int(assignment[c2])
                if f1 == f2:
                    continue
                load[f1] -= demands[c1]
                load[f2] -= demands[c2]
                if load[f1] + demands[c2] > capacities[f1] + 1e-9:
                    load[f1] += demands[c1]
                    load[f2] += demands[c2]
                    continue
                if load[f2] + demands[c1] > capacities[f2] + 1e-9:
                    load[f1] += demands[c1]
                    load[f2] += demands[c2]
                    continue
                delta = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                        (assign_cost[c2, f1] - assign_cost[c2, f2])
                if load[f1] == 0:
                    delta -= open_cost[f1]
                if load[f2] == 0:
                    delta -= open_cost[f2]
                if delta < -1e-9:
                    assignment[c1] = f2
                    assignment[c2] = f1
                    load[f1] += demands[c2]
                    load[f2] += demands[c1]
                    current_cost += delta
                    improved = True
                    swap_improved = True
                else:
                    load[f1] += demands[c1]
                    load[f2] += demands[c2]

            if not improved and not swap_improved:
                break

        return assignment

    best_assignment = None
    best_cost = np.inf

    # --- Phase 1: Try MIP ---
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds

        n_vars = n_customers * n_facilities + n_facilities
        x_vars = n_customers * n_facilities

        c_obj = np.zeros(n_vars)
        for c in range(n_customers):
            for f in range(n_facilities):
                c_obj[c * n_facilities + f] = assign_cost[c, f]
        for f in range(n_facilities):
            c_obj[x_vars + f] = open_cost[f]

        n_cons = n_customers + n_facilities + n_customers * n_facilities
        A = np.zeros((n_cons, n_vars))
        lb = np.zeros(n_cons)
        ub = np.zeros(n_cons)

        for c in range(n_customers):
            A[c, c * n_facilities:(c + 1) * n_facilities] = 1.0
            lb[c] = 1.0
            ub[c] = 1.0

        for f in range(n_facilities):
            row = n_customers + f
            for c in range(n_customers):
                A[row, c * n_facilities + f] = demands[c]
            A[row, x_vars + f] = -capacities[f]
            lb[row] = -np.inf
            ub[row] = 0.0

        for c in range(n_customers):
            for f in range(n_facilities):
                row = n_customers + n_facilities + c * n_facilities + f
                A[row, c * n_facilities + f] = 1.0
                A[row, x_vars + f] = -1.0
                lb[row] = -np.inf
                ub[row] = 0.0

        constraints = LinearConstraint(A, lb, ub)
        var_bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)

        result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=integrality,
            bounds=var_bounds,
            options={'time_limit': 60}
        )

        if result.status == 0 and result.x is not None:
            assignment = np.zeros(n_customers, dtype=int)
            for c in range(n_customers):
                for f in range(n_facilities):
                    if result.x[c * n_facilities + f] > 0.5:
                        assignment[c] = f
                        break
            cost = compute_cost(assignment)
            if cost < np.inf:
                best_assignment = assignment.copy()
                best_cost = cost

    except Exception:
        pass

    # --- Phase 2: LP relaxation + rounding (if MIP didn't give a good solution) ---
    if time.time() < deadline - 40:
        try:
            from scipy.optimize import linprog, LinearConstraint, Bounds

            n_vars = n_customers * n_facilities + n_facilities
            x_vars = n_customers * n_facilities

            c_obj = np.zeros(n_vars)
            for c in range(n_customers):
                for f in range(n_facilities):
                    c_obj[c * n_facilities + f] = assign_cost[c, f]
            for f in range(n_facilities):
                c_obj[x_vars + f] = open_cost[f]

            n_cons = n_customers + n_facilities + n_customers * n_facilities
            A = np.zeros((n_cons, n_vars))
            lb = np.zeros(n_cons)
            ub = np.zeros(n_cons)

            for c in range(n_customers):
                A[c, c * n_facilities:(c + 1) * n_facilities] = 1.0
                lb[c] = 1.0
                ub[c] = 1.0

            for f in range(n_facilities):
                row = n_customers + f
                for c in range(n_customers):
                    A[row, c * n_facilities + f] = demands[c]
                A[row, x_vars + f] = -capacities[f]
                lb[row] = -np.inf
                ub[row] = 0.0

            for c in range(n_customers):
                for f in range(n_facilities):
                    row = n_customers + n_facilities + c * n_facilities + f
                    A[row, c * n_facilities + f] = 1.0
                    A[row, x_vars + f] = -1.0
                    lb[row] = -np.inf
                    ub[row] = 0.0

            constraints = LinearConstraint(A, lb, ub)
            var_bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

            result = linprog(
                c=c_obj,
                A_ub=A,
                b_ub=ub,
                A_eq=None,
                b_eq=None,
                bounds=var_bounds,
                method='highs',
                options={'time_limit': 30}
            )

            if result.success and result.x is not None:
                # Round LP solution: for each customer, pick facility with highest x value
                lp_solution = result.x[:x_vars].reshape(n_customers, n_facilities)
                assignment = np.zeros(n_customers, dtype=int)
                for c in range(n_customers):
                    assignment[c] = int(np.argmax(lp_solution[c]))

                # Fix infeasibilities: if capacity violated, reassign overflow customers
                load = np.zeros(n_facilities)
                for c in range(n_customers):
                    load[assignment[c]] += demands[c]

                # Fix capacity violations
                for f in range(n_facilities):
                    overflow = load[f] - capacities[f]
                    if overflow > 1e-9:
                        # Find customers at f with highest cost delta to move
                        custs_at_f = [c for c in range(n_customers) if assignment[c] == f]
                        custs_at_f.sort(key=lambda c: -demands[c])
                        for c in custs_at_f:
                            if load[f] - demands[c] <= capacities[f]:
                                break
                            # Move c to best feasible facility
                            best_f2 = -1
                            best_cost_move = np.inf
                            for f2 in range(n_facilities):
                                if f2 == f:
                                    continue
                                if load[f2] + demands[c] <= capacities[f2] + 1e-9:
                                    cost_move = assign_cost[c, f2] - assign_cost[c, f]
                                    if cost_move < best_cost_move:
                                        best_cost_move = cost_move
                                        best_f2 = f2
                            if best_f2 >= 0:
                                load[f] -= demands[c]
                                load[best_f2] += demands[c]
                                assignment[c] = best_f2

                cost = compute_cost(assignment)
                if cost < np.inf:
                    # Polish with local search
                    assignment = local_search(assignment)
                    cost = compute_cost(assignment)
                    if cost < best_cost:
                        best_cost = cost
                        best_assignment = assignment.copy()

        except Exception:
            pass

    # --- Phase 3: Greedy + local search fallback ---
    if time.time() < deadline - 15:
        orderings = [
            np.arange(n_customers),
            np.argsort(-demands),
            np.argsort(demands),
            np.argsort(assign_cost.sum(axis=1)),
            np.argsort(assign_cost.min(axis=1)),
        ]

        for order in orderings:
            if time.time() > deadline - 15:
                break
            sol = greedy_assignment(order)
            if sol is not None:
                sol = local_search(sol)
                cost = compute_cost(sol)
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = sol.copy()

        # Randomized greedy
        rng = np.random.RandomState(42)
        n_random = 15
        for i in range(n_random):
            if time.time() > deadline - 10:
                break
            order = rng.permutation(n_customers)
            sol = greedy_assignment(order)
            if sol is not None:
                sol = local_search(sol)
                cost = compute_cost(sol)
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = sol.copy()

    # --- Final: ensure we have a valid assignment ---
    if best_assignment is None:
        # Last resort: assign to facility with most remaining capacity
        best_assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(np.argmax(capacities - load))
            best_assignment[c] = f
            load[f] += demands[c]

    return best_assignment.astype(int)
# EVOLVE-BLOCK-END
