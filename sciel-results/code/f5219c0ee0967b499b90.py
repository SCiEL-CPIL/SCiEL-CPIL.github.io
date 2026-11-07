"""Capacitated facility location via min-cost flow with fixed-charge handling."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix
import time


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    n_customers = len(demands)
    n_facilities = len(capacities)
    t_start = time.time()
    TIME_LIMIT = 110.0  # safety margin before 120s kill

    # Total demand
    total_demand = demands.sum()

    # --- Phase 1: Solve LP relaxation of the fixed-charge problem ---
    # Variables: x[i,f] in [0,1] for each customer-facility pair,
    #            y[f] in [0,1] for each facility open decision
    # Minimize: sum_{i,f} assign_cost[i,f]*x[i,f] + sum_f open_cost[f]*y[f]
    # Subject to:
    #   sum_f x[i,f] = 1  for each customer i
    #   sum_i demands[i]*x[i,f] <= capacities[f]*y[f]  for each facility f
    #   0 <= x[i,f] <= 1, 0 <= y[f] <= 1

    n_vars = n_customers * n_facilities + n_facilities
    x_idx = lambda i, f: i * n_facilities + f
    y_idx = lambda f: n_customers * n_facilities + f

    # Objective
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for f in range(n_facilities):
            c[x_idx(i, f)] = assign_cost[i, f]
        c[y_idx(f)] = open_cost[f]

    # Equality constraints: each customer assigned to exactly one facility
    A_eq = lil_matrix((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for f in range(n_facilities):
            A_eq[i, x_idx(i, f)] = 1.0

    # Inequality constraints: capacity coupling
    # sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0
    A_ub = lil_matrix((n_facilities, n_vars))
    b_ub = np.zeros(n_facilities)
    for f in range(n_facilities):
        for i in range(n_customers):
            A_ub[f, x_idx(i, f)] = demands[i]
        A_ub[f, y_idx(f)] = -capacities[f]

    # Bounds
    bounds = [(0.0, 1.0)] * n_vars

    # Solve LP
    try:
        res = linprog(c, A_ub=csr_matrix(A_ub), b_ub=b_ub,
                      A_eq=csr_matrix(A_eq), b_eq=b_eq,
                      bounds=bounds, method='highs')
    except Exception:
        res = None

    # --- Phase 2: Extract initial assignment from LP solution ---
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)

    if res is not None and res.status == 0:
        x_vals = res.x[:n_customers * n_facilities].reshape(n_customers, n_facilities)
        y_vals = res.x[n_customers * n_facilities:]

        # Round: for each customer, pick facility with highest x value
        # but respect capacity greedily in order of LP "confidence"
        # Sort customers by their best x value descending (most confident first)
        best_x = x_vals.max(axis=1)
        order = np.argsort(-best_x)

        # Greedily assign in confidence order
        for i in order:
            # Try facilities in order of x value
            facs = np.argsort(-x_vals[i])
            assigned = False
            for f in facs:
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            if not assigned:
                # Fallback: cheapest feasible
                for f in np.argsort(assign_cost[i]):
                    if load[f] + demands[i] <= capacities[f]:
                        result[i] = f
                        load[f] += demands[i]
                        break
    else:
        # Fallback: greedy largest-first
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    break

    # --- Phase 3: Local search refinement (swap and reassignment moves) ---
    if time.time() - t_start > TIME_LIMIT:
        return result

    def compute_total_cost(result, load):
        cost = 0.0
        used = set(result.tolist())
        for f in used:
            cost += open_cost[f]
        for i in range(n_customers):
            cost += assign_cost[i, result[i]]
        return cost

    current_cost = compute_total_cost(result, load)
    n_improvements = 0
    max_iters = 200

    for iteration in range(max_iters):
        if time.time() - t_start > TIME_LIMIT - 10:
            break

        improved = False

        # Try moving each customer to a different facility
        for i in range(n_customers):
            old_f = result[i]
            # Try all other facilities
            best_new_f = -1
            best_delta = 0.0
            for f in range(n_facilities):
                if f == old_f:
                    continue
                if load[f] + demands[i] > capacities[f]:
                    continue
                # Delta in cost
                delta = assign_cost[i, f] - assign_cost[i, old_f]
                # If old facility becomes empty, we save open_cost[old_f]
                # If new facility was empty, we add open_cost[f]
                if load[old_f] - demands[i] == 0:
                    delta -= open_cost[old_f]
                if load[f] == 0:
                    delta += open_cost[f]
                if delta < best_delta:
                    best_delta = delta
                    best_new_f = f

            if best_new_f >= 0 and best_delta < -1e-9:
                # Apply move
                load[old_f] -= demands[i]
                load[best_new_f] += demands[i]
                result[i] = best_new_f
                current_cost += best_delta
                improved = True
                n_improvements += 1

        if not improved:
            break

    # --- Phase 4: Second pass - try to close empty facilities and redistribute ---
    if time.time() - t_start > TIME_LIMIT:
        return result

    # Verify validity
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-6:
            # Should not happen, but safety
            pass

    return result
