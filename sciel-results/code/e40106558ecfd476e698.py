# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s limit

    # 1. Parse Inputs
    if isinstance(distribution_spec, dict):
        if 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.array([distribution_spec.get(i, 0.0) for i in range(16)], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0

    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    probs = np.clip(probs, 0, 1)

    if isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i // 4, i % 4) for i in range(16)]

    if len(support_list) != 16:
        support_list = [(i // 4, i % 4) for i in range(16)]

    V = np.zeros((16, 2))
    for i, (v1, v2) in enumerate(support_list):
        V[i, 0] = v1
        V[i, 1] = v2

    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)
    best_revenue = 0.0

    def check_validity(allocations, payments, V, tol=1e-8):
        """Check IR and IC constraints"""
        for i in range(16):
            util = np.dot(V[i], allocations[i]) - payments[i]
            if util < -tol:
                return False
        for i in range(16):
            for ip in range(16):
                util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                util_truth = np.dot(V[i], allocations[i]) - payments[i]
                if util_report - util_truth > tol:
                    return False
        return True

    def extract_solution(x, V):
        """Extract allocations and payments from LP solution vector"""
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(16):
            allocations[i, 0] = x[2*i]
            allocations[i, 1] = x[2*i+1]
            payments[i] = x[32+i]
        allocations = np.clip(allocations, 0, 1)
        payments = np.maximum(payments, 0)
        return allocations, payments

    def build_ic_lp(V, probs, tol=1e-10):
        """Build the 48-var IC LP"""
        n_vars = 48
        c = np.zeros(n_vars)
        for i in range(16):
            c[32 + i] = -probs[i]

        A_ir = np.zeros((16, n_vars))
        b_ir = np.zeros(16)
        for i in range(16):
            A_ir[i, 2*i] = -V[i, 0]
            A_ir[i, 2*i+1] = -V[i, 1]
            A_ir[i, 32+i] = 1

        n_ic = 16 * 16
        A_ic = np.zeros((n_ic, n_vars))
        b_ic = np.zeros(n_ic)

        for i in range(16):
            for ip in range(16):
                idx = i * 16 + ip
                A_ic[idx, 2*ip] += V[i, 0]
                A_ic[idx, 2*ip+1] += V[i, 1]
                A_ic[idx, 32+ip] += -1
                A_ic[idx, 2*i] += -V[i, 0]
                A_ic[idx, 2*i+1] += -V[i, 1]
                A_ic[idx, 32+i] += 1

        A = np.vstack([A_ir, A_ic])
        b = np.concatenate([b_ir, b_ic])

        bounds = []
        for i in range(16):
            bounds.append((0, 1))
            bounds.append((0, 1))
        for i in range(16):
            bounds.append((0, None))

        return c, A, b, bounds, tol

    # --- Approach 3: Utility-based formulation with explicit allocations ---
    # Variables: u_i (16) + a_i1, a_i2 (32) = 48 vars
    # Objective: Maximize Expected Revenue = Sum p_i * (v_i . a_i - u_i)
    # Minimize: -Sum p_i * (v_i . a_i - u_i)
    
    n_vars = 48
    c = np.zeros(n_vars)
    for i in range(16):
        c[2*i] = -probs[i] * V[i, 0]
        c[2*i+1] = -probs[i] * V[i, 1]
        c[32+i] = probs[i]

    A_ub_rows = []
    b_ub_vals = []

    # IR: u_i >= 0 => -u_i <= 0
    for i in range(16):
        row = np.zeros(n_vars)
        row[32+i] = -1
        A_ub_rows.append(row)
        b_ub_vals.append(0)

    # IC: u_i - u_j >= (v_i - v_j) . a_j
    # => (v_i - v_j) . a_j - u_i + u_j <= 0
    for i in range(16):
        for j in range(16):
            row = np.zeros(n_vars)
            row[2*j] = V[i, 0] - V[j, 0]
            row[2*j+1] = V[i, 1] - V[j, 1]
            row[32+i] = -1
            row[32+j] = 1
            A_ub_rows.append(row)
            b_ub_vals.append(0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)

    bounds = []
    for i in range(16):
        bounds.append((0, 1))
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))

    # Try solving with high precision first
    tolerances = [1e-12, 1e-10, 1e-8]
    
    for tol in tolerances:
        if time.time() > deadline - 2.0:
            break
        try:
            options = {
                'presolve': True,
                'dual_feasibility_tolerance': tol,
                'primal_feasibility_tolerance': tol,
                'dual_infeasibility_tolerance': tol,
                'primal_infeasibility_tolerance': tol,
                'ipm_optimality_tolerance': tol,
                'time_limit': 20.0
            }
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)

            if result.success:
                x = result.x
                allocations = np.zeros((16, 2))
                payments = np.zeros(16)
                for i in range(16):
                    allocations[i, 0] = x[2*i]
                    allocations[i, 1] = x[2*i+1]
                    u_i = x[32+i]
                    payments[i] = np.dot(V[i], allocations[i]) - u_i

                allocations = np.clip(allocations, 0, 1)
                payments = np.maximum(payments, 0)

                if check_validity(allocations, payments, V):
                    revenue = np.dot(probs, payments)
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_allocations = allocations.copy()
                        best_payments = payments.copy()
                        # If we found a good solution with tight tolerance, we can stop
                        if tol <= 1e-12:
                            break
        except Exception:
            pass

    return {'allocations': best_allocations, 'payments': best_payments}
# EVOLVE-BLOCK-END
