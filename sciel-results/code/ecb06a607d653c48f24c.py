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
        # IR: v_i . x_i - p_i >= 0
        for i in range(16):
            util = np.dot(V[i], allocations[i]) - payments[i]
            if util < -tol:
                return False
        # IC: v_i . x_j - p_j <= v_i . x_i - p_i
        for i in range(16):
            util_truth = np.dot(V[i], allocations[i]) - payments[i]
            for j in range(16):
                util_report = np.dot(V[i], allocations[j]) - payments[j]
                if util_report - util_truth > tol:
                    return False
        return True

    def extract_solution_u(x, V):
        """Extract allocations and payments from utility-based LP solution vector"""
        # x[0:32] are allocations, x[32:48] are utilities
        allocations = np.zeros((16, 2))
        utilities = np.zeros(16)
        for i in range(16):
            allocations[i, 0] = x[2*i]
            allocations[i, 1] = x[2*i+1]
            utilities[i] = x[32+i]
        
        payments = np.zeros(16)
        for i in range(16):
            payments[i] = np.dot(V[i], allocations[i]) - utilities[i]
            
        allocations = np.clip(allocations, 0, 1)
        payments = np.maximum(payments, 0)
        return allocations, payments

    # --- Approach 1: Utility-based LP Formulation ---
    # Variables: x_i1, x_i2, u_i for i=0..15
    # Total 48 variables
    # Objective: min sum p_i * (u_i - v_i . x_i)
    # c[i] for x_i1: -p_i * V[i,0]
    # c[i] for x_i2: -p_i * V[i,1]
    # c[i] for u_i: p_i
    
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

    # IC: u_i - u_j >= v_i . (x_j - x_i)
    # => v_i . x_j - v_i . x_i - u_i + u_j <= 0
    for i in range(16):
        for j in range(16):
            if i == j:
                continue # Trivially satisfied, skip to reduce constraints
            row = np.zeros(n_vars)
            # v_i . x_j
            row[2*j] += V[i, 0]
            row[2*j+1] += V[i, 1]
            # - v_i . x_i
            row[2*i] -= V[i, 0]
            row[2*i+1] -= V[i, 1]
            # - u_i
            row[32+i] -= 1
            # + u_j
            row[32+j] += 1
            
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

    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'dual_infeasibility_tolerance': 1e-12,
            'primal_infeasibility_tolerance': 1e-12,
            'ipm_optimality_tolerance': 1e-12,
            'time_limit': 20.0
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
        
        if result.success:
            allocations, payments = extract_solution_u(result.x, V)
            if check_validity(allocations, payments, V):
                revenue = np.dot(probs, payments)
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_allocations = allocations.copy()
                    best_payments = payments.copy()
            else:
                # If slight numerical violation, try to repair
                # Increase payments slightly if IR violated? No, IR is p <= v.x.
                # If IC violated, it's harder.
                # Let's try the direct P,X formulation as a backup
                pass
    except Exception:
        pass

    if time.time() > deadline - 5.0:
        return {'allocations': best_allocations, 'payments': best_payments}

    # --- Approach 2: Direct P, X Formulation (Backup/Refinement) ---
    # Variables: x_i1, x_i2, p_i
    # Objective: min - sum p_i * p_i_prob
    n_vars = 48
    c = np.zeros(n_vars)
    for i in range(16):
        c[32+i] = -probs[i]

    A_ub_rows = []
    b_ub_vals = []

    # IR: v_i . x_i - p_i >= 0 => -v_i . x_i + p_i <= 0
    for i in range(16):
        row = np.zeros(n_vars)
        row[2*i] = -V[i, 0]
        row[2*i+1] = -V[i, 1]
        row[32+i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0)

    # IC: v_i . x_j - p_j - v_i . x_i + p_i <= 0
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[2*j] = V[i, 0]
            row[2*j+1] = V[i, 1]
            row[32+j] = -1
            row[2*i] = -V[i, 0]
            row[2*i+1] = -V[i, 1]
            row[32+i] = 1
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

    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'dual_infeasibility_tolerance': 1e-12,
            'primal_infeasibility_tolerance': 1e-12,
            'ipm_optimality_tolerance': 1e-12,
            'time_limit': 20.0
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
        
        if result.success:
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            for i in range(16):
                allocations[i, 0] = result.x[2*i]
                allocations[i, 1] = result.x[2*i+1]
                payments[i] = result.x[32+i]
            
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)
            
            if check_validity(allocations, payments, V):
                revenue = np.dot(probs, payments)
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_allocations = allocations.copy()
                    best_payments = payments.copy()
    except Exception:
        pass

    # --- Polish Step: If we have a solution, check if we can tighten it ---
    # The LP is convex, so the first valid solution is optimal.
    # However, if the first one failed validity check due to 1e-9 error,
    # we might have a second one.
    
    return {'allocations': best_allocations, 'payments': best_payments}
