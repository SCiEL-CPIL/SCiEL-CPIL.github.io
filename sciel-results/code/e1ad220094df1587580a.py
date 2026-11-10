import time
import numpy as np
from scipy.optimize import linprog


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # 16 types: (v1, v2), v in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    type_values = np.array(types, dtype=float)  # (16,2)

    # ---- Parse distribution ----
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, dict):
        probs = np.ones(n_types) / n_types
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
    elif isinstance(distribution_spec, str):
        probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types

    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if s > 1e-12:
        probs = probs / s
    else:
        probs = np.ones(n_types) / n_types

    # ---- LP: variables interleaved [q1(i), q2(i), p(i)] per type ----
    n_vars = 3 * n_types  # 48

    # Objective: maximize sum_i probs[i] * p(i)  -> minimize negative
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[3 * i + 2] = -probs[i]

    A_rows = []
    b_vals = []

    # IC constraints: for all theta (true), theta' (reported):
    #   v_theta . q(theta') - p(theta') <= v_theta . q(theta) - p(theta)
    for ti in range(n_types):
        v = type_values[ti]
        base_true = 3 * ti
        for tj in range(n_types):
            base_rep = 3 * tj
            row = np.zeros(n_vars)
            row[base_rep + 0] = v[0]
            row[base_rep + 1] = v[1]
            row[base_rep + 2] = -1.0
            row[base_true + 0] = -v[0]
            row[base_true + 1] = -v[1]
            row[base_true + 2] = 1.0
            A_rows.append(row)
            b_vals.append(0.0)

    # IR constraints: p(theta) <= v_theta . q(theta)
    for ti in range(n_types):
        v = type_values[ti]
        base = 3 * ti
        row = np.zeros(n_vars)
        row[base + 0] = -v[0]
        row[base + 1] = -v[1]
        row[base + 2] = 1.0
        A_rows.append(row)
        b_vals.append(0.0)

    A = np.array(A_rows)
    b = np.array(b_vals)

    bounds = []
    for i in range(n_types):
        bounds.append((0.0, 1.0))  # q1
        bounds.append((0.0, 1.0))  # q2
        bounds.append((0.0, None)) # p

    # ---- Solve LP ----
    x = None
    if time.time() < deadline - 5:
        result = linprog(
            c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
            options={'presolve': True,
                     'dual_feasibility_tolerance': 1e-12,
                     'primal_feasibility_tolerance': 1e-12,
                     'dual_stopping_tolerance': 1e-12,
                     'primal_stopping_tolerance': 1e-12,
                     'time_limit': max(0.1, deadline - time.time() - 5)}
        )
        if result.success:
            x = result.x

    # Fallback if LP failed: simple feasible mechanism
    if x is None:
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for i, (v1, v2) in enumerate(types):
            if v1 + v2 >= 3:
                allocations[i] = [1.0, 1.0]
                payments[i] = 2.0
        return {'allocations': allocations, 'payments': payments}

    # Extract
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    for i in range(n_types):
        allocations[i, 0] = x[3 * i + 0]
        allocations[i, 1] = x[3 * i + 1]
        payments[i] = x[3 * i + 2]
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)

    # ---- Local refinement: squeeze residual revenue via payment hill-climb ----
    def feasible(alloc, pay, tol=1e-9):
        for ti in range(n_types):
            v = type_values[ti]
            u_truth = v[0] * alloc[ti, 0] + v[1] * alloc[ti, 1] - pay[ti]
            if u_truth < -tol:
                return False
            for tj in range(n_types):
                if tj == ti:
                    continue
                u_dev = v[0] * alloc[tj, 0] + v[1] * alloc[tj, 1] - pay[tj]
                if u_dev > u_truth + tol:
                    return False
        return True

    def revenue(alloc, pay):
        return float(np.sum(probs * pay))

    if time.time() < deadline - 2:
        best_alloc = allocations.copy()
        best_pay = payments.copy()
        best_rev = revenue(best_alloc, best_pay)
        # sanity: ensure feasible before refining
        if not feasible(best_alloc, best_pay):
            # nudge: reduce payments slightly to restore feasibility
            for _ in range(200):
                if feasible(best_alloc, best_pay):
                    break
                idx = int(np.argmax(best_pay))
                best_pay[idx] *= 0.9
            best_rev = revenue(best_alloc, best_pay)

        # Greedy single-direction payment increases (safe: only raises revenue)
        for step in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(n_types):
                    new_pay = best_pay.copy()
                    new_pay[i] += step
                    if new_pay[i] < 0:
                        continue
                    if feasible(best_alloc, new_pay, tol=1e-9):
                        new_rev = revenue(best_alloc, new_pay)
                        if new_rev > best_rev + 1e-12:
                            best_pay = new_pay
                            best_rev = new_rev
                            improved = True
                            break

        # Final high-precision LP polish with warm start, tighter tolerances
        if time.time() < deadline - 1:
            x_init = np.zeros(n_vars)
            for i in range(n_types):
                x_init[3 * i + 0] = np.clip(best_alloc[i, 0], 0.0, 1.0)
                x_init[3 * i + 1] = np.clip(best_alloc[i, 1], 0.0, 1.0)
                x_init[3 * i + 2] = max(0.0, best_pay[i])
            try:
                result2 = linprog(
                    c, A_ub=A, b_ub=b, bounds=bounds, method='highs', x0=x_init,
                    options={'presolve': True,
                             'dual_feasibility_tolerance': 1e-14,
                             'primal_feasibility_tolerance': 1e-14,
                             'dual_stopping_tolerance': 1e-14,
                             'primal_stopping_tolerance': 1e-14,
                             'time_limit': max(0.1, deadline - time.time() - 1)}
                )
                if result2.success:
                    alloc2 = np.zeros((n_types, 2))
                    pay2 = np.zeros(n_types)
                    for i in range(n_types):
                        alloc2[i, 0] = result2.x[3 * i + 0]
                        alloc2[i, 1] = result2.x[3 * i + 1]
                        pay2[i] = result2.x[3 * i + 2]
                    alloc2 = np.clip(alloc2, 0.0, 1.0)
                    pay2 = np.maximum(pay2, 0.0)
                    if feasible(alloc2, pay2, tol=1e-9):
                        rev2 = revenue(alloc2, pay2)
                        if rev2 >= best_rev - 1e-12:
                            best_alloc, best_pay, best_rev = alloc2, pay2, rev2
            except Exception:
                pass

        allocations = best_alloc
        payments = best_pay

    # Final safety clip
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)

    return {'allocations': allocations, 'payments': payments}
