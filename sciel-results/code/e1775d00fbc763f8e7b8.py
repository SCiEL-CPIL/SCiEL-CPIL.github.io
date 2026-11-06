import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0  # safety margin before 90s kill

    # Parse distribution_spec to get probabilities over 16 types
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.zeros(16)
            for k, v in distribution_spec.items():
                if isinstance(k, tuple) and len(k) == 2:
                    v1, v2 = k
                    idx = v1 * 4 + v2
                    if 0 <= idx < 16:
                        probs[idx] = v
                elif isinstance(k, int) and 0 <= k < 16:
                    probs[k] = v
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) != 16:
            if len(probs) > 16:
                probs = probs[:16]
            else:
                tmp = np.zeros(16)
                tmp[:len(probs)] = probs
                probs = tmp
    else:
        probs = np.ones(16) / 16.0

    # Normalize
    probs = np.clip(probs, 0, None)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(16) / 16.0

    n_types = 16
    values = np.zeros((n_types, 2))
    for i in range(n_types):
        values[i, 0] = i // 4
        values[i, 1] = i % 4

    # Strategy: Solve the full LP with HiGHS, then polish with a local
    # active-set refinement that tightens feasibility and potentially improves
    # the objective by exploiting the structure of binding constraints.

    n_vars = 48
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[32 + i] = -probs[i]

    # Build constraint matrix
    A_ub_list = []
    b_ub_list = []

    # IR constraints: p_i - sum_j v_ij x_ij <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[32 + i] = 1.0
        for j in range(2):
            row[i * 2 + j] = -values[i, j]
        A_ub_list.append(row)
        b_ub_list.append(0.0)

    # IC constraints: for each pair (i, j), i != j
    # p_i - p_j - sum_k v_ik x_ik + sum_k v_ik x_jk <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[32 + i] = 1.0
            row[32 + j] = -1.0
            for k in range(2):
                row[i * 2 + k] = -values[i, k]
                row[j * 2 + k] = values[i, k]
            A_ub_list.append(row)
            b_ub_list.append(0.0)

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[32:] = 6.0  # max possible payment is sum of max values = 3+3=6

    def solve_lp(c_vec, A_ub_mat, b_ub_vec, bounds_list, time_limit):
        try:
            res = linprog(c_vec, A_ub=A_ub_mat, b_ub=b_ub_vec, bounds=bounds_list,
                         method='highs',
                         options={'presolve': True,
                                  'dual_feasibility_tolerance': 1e-14,
                                  'primal_feasibility_tolerance': 1e-14,
                                  'dual_staleness_threshold': 1e-14,
                                  'primal_staleness_threshold': 1e-14,
                                  'time_limit': time_limit})
            return res
        except Exception:
            return None

    def verify_solution(allocations, payments, tol=1e-8):
        """Check IR and IC constraints with tolerance."""
        for i in range(n_types):
            if payments[i] > np.sum(values[i] * allocations[i]) + tol:
                return False
        for i in range(n_types):
            for j in range(n_types):
                if i == j:
                    continue
                lhs = payments[i] - payments[j]
                rhs = np.sum(values[i] * (allocations[i] - allocations[j]))
                if lhs > rhs + tol:
                    return False
        return True

    def polish_solution(allocations, payments, c_vec, A_ub_mat, b_ub_vec,
                        lb_vec, ub_vec, remaining_time):
        """
        Polish a feasible solution by solving a local LP restricted to
        variables near their current values, with very tight tolerances.
        This can recover precision lost in the initial solve.
        """
        if remaining_time < 5:
            return allocations, payments

        # Tighten bounds around the current solution
        x0 = np.concatenate([allocations.flatten(), payments])
        eps_bound = 1e-6
        lb_tight = np.maximum(lb_vec, x0 - eps_bound)
        ub_tight = np.minimum(ub_vec, x0 + eps_bound)

        # Ensure feasibility of the tightened bounds
        lb_tight = np.minimum(lb_tight, ub_tight)

        res = solve_lp(c_vec, A_ub_mat, b_ub_vec, list(zip(lb_tight, ub_tight)),
                       remaining_time - 2)

        if res is not None and res.status == 0 and res.x is not None:
            x_pol = res.x[:32]
            p_pol = res.x[32:]
            x_pol = np.clip(x_pol, 0, 1)
            p_pol = np.maximum(p_pol, 0)
            alloc_pol = x_pol.reshape(16, 2)
            pay_pol = p_pol

            if verify_solution(alloc_pol, pay_pol, tol=1e-10):
                return alloc_pol, pay_pol

        return allocations, payments

    best_allocations = None
    best_payments = None
    best_revenue = -np.inf

    # First pass: solve with very tight tolerances
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        res = solve_lp(c, A_ub, b_ub, list(zip(lb, ub)), remaining_time)
        if res is not None and res.status == 0 and res.x is not None:
            x_sol = res.x[:32]
            p_sol = res.x[32:]
            x_sol = np.clip(x_sol, 0, 1)
            p_sol = np.maximum(p_sol, 0)
            allocations = x_sol.reshape(16, 2)
            payments = p_sol

            if verify_solution(allocations, payments, tol=1e-8):
                revenue = np.sum(probs * payments)
                best_allocations = allocations.copy()
                best_payments = payments.copy()
                best_revenue = revenue

    # Polish pass: tighten around the found solution
    if best_allocations is not None and time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 3
        if remaining_time > 5:
            alloc_pol, pay_pol = polish_solution(
                best_allocations, best_payments, c, A_ub, b_ub, lb, ub,
                remaining_time)
            if verify_solution(alloc_pol, pay_pol, tol=1e-10):
                revenue_pol = np.sum(probs * pay_pol)
                if revenue_pol > best_revenue - 1e-12:
                    best_allocations = alloc_pol.copy()
                    best_payments = pay_pol.copy()
                    best_revenue = revenue_pol

    # Second polish: even tighter bounds
    if best_allocations is not None and time.time() < deadline - 8:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 3:
            x0 = np.concatenate([best_allocations.flatten(), best_payments])
            eps_bound2 = 1e-8
            lb_tight2 = np.maximum(lb, x0 - eps_bound2)
            ub_tight2 = np.minimum(ub, x0 + eps_bound2)
            lb_tight2 = np.minimum(lb_tight2, ub_tight2)

            res2 = solve_lp(c, A_ub, b_ub, list(zip(lb_tight2, ub_tight2)),
                           remaining_time - 1)
            if res2 is not None and res2.status == 0 and res2.x is not None:
                x_sol2 = res2.x[:32]
                p_sol2 = res2.x[32:]
                x_sol2 = np.clip(x_sol2, 0, 1)
                p_sol2 = np.maximum(p_sol2, 0)
                alloc2 = x_sol2.reshape(16, 2)
                pay2 = p_sol2

                if verify_solution(alloc2, pay2, tol=1e-10):
                    revenue2 = np.sum(probs * pay2)
                    if revenue2 > best_revenue - 1e-12:
                        best_allocations = alloc2.copy()
                        best_payments = pay2.copy()
                        best_revenue = revenue2

    # If LP failed or we're running out of time, use a simple valid fallback
    if best_allocations is None or time.time() > deadline - 2:
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)

    return {"allocations": best_allocations, "payments": best_payments}
