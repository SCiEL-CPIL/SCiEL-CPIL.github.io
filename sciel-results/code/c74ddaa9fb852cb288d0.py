import numpy as np
from itertools import product
from scipy.optimize import linprog
import time
import math

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    timeout = 50.0  # Safety margin before 90s kill

    values = [0, 1, 2, 3]
    types = [(i, j) for i in values for j in values]
    n_types = 16
    type_to_vals = {i: t for i, t in enumerate(types)}

    # Get distribution
    probs = None
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            elif 'values' in distribution_spec and 'probs' in distribution_spec:
                vals = np.array(distribution_spec['values'])
                ps = np.array(distribution_spec['probs'])
                if len(vals) == 16 and len(ps) == 16:
                    probs = ps
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    n = 16
    
    # Phase 1: Solve the full LP for allocations and payments
    c = np.zeros(3 * n)
    for t in range(n):
        c[32 + t] = -probs[t]

    A_rows = []
    b_rows = []
    
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(3 * n)
            row[0 + t] = -v1_t
            row[16 + t] = -v2_t
            row[0 + t2] = v1_t
            row[16 + t2] = v2_t
            row[32 + t] = 1
            row[32 + t2] = -1
            A_rows.append(row)
            b_rows.append(0)
            
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        row = np.zeros(3 * n)
        row[0 + t] = -v1_t
        row[16 + t] = -v2_t
        row[32 + t] = 1
        A_rows.append(row)
        b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = []
    for _ in range(n):
        bounds.append((0, 1))
    for _ in range(n):
        bounds.append((0, 1))
    for _ in range(n):
        bounds.append((0, None))

    best_result = None
    best_obj = -np.inf
    
    methods = ['highs', 'interior-point']
    
    for method in methods:
        if time.time() - start_time > timeout:
            break
            
        try:
            if method == 'highs':
                options = {'presolve': True, 'dual_feasibility_tolerance': 1e-14, 'primal_feasibility_tolerance': 1e-14, 'dual_gap_tolerance': 1e-14}
            else:
                options = {'maxiter': 5000, 'tol': 1e-14}
                
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method, options=options)
            
            if result.success:
                obj = -result.fun
                if obj > best_obj:
                    best_obj = obj
                    best_result = result
                    
        except Exception:
            pass

    if best_result is None:
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}

    x = best_result.x
    x1 = x[0:16]
    x2 = x[16:32]
    
    allocations = np.column_stack((x1, x2))
    allocations = np.clip(allocations, 0, 1)
    
    # Phase 2: Polish - recompute optimal payments given the allocation
    # For fixed allocation x, find max sum_t probs[t]*p(t) subject to IC and IR
    c_pay = -probs.copy()
    
    A_pay_rows = []
    b_pay_rows = []
    
    # IC constraints: p(t) - p(t') <= v(t).x(t) - v(t).x(t')
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        vx_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        for t2 in range(n):
            if t == t2:
                continue
            v1_t2, v2_t2 = type_to_vals[t2]
            vx_t2 = v1_t2 * allocations[t2, 0] + v2_t2 * allocations[t2, 1]
            row = np.zeros(n)
            row[t] = 1
            row[t2] = -1
            A_pay_rows.append(row)
            b_pay_rows.append(vx_t - vx_t2)
    
    # IR constraints: p(t) <= v(t).x(t)
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        vx_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        row = np.zeros(n)
        row[t] = 1
        A_pay_rows.append(row)
        b_pay_rows.append(vx_t)
    
    A_pay_ub = np.array(A_pay_rows)
    b_pay_ub = np.array(b_pay_rows)
    
    pay_bounds = [(0, None)] * n
    
    try:
        pay_result = linprog(c_pay, A_ub=A_pay_ub, b_ub=b_pay_ub, bounds=pay_bounds, method='highs',
                            options={'presolve': True, 'dual_feasibility_tolerance': 1e-14, 
                                     'primal_feasibility_tolerance': 1e-14, 'dual_gap_tolerance': 1e-14})
        if pay_result.success:
            payments = pay_result.x
            payments = np.maximum(payments, 0)
        else:
            # Fallback to original payments
            payments = x[32:48]
            payments = np.maximum(payments, 0)
    except Exception:
        payments = x[32:48]
        payments = np.maximum(payments, 0)
    
    # Final repair: ensure IR and IC hold
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        if payments[t] > val_t:
            payments[t] = val_t - 1e-12

    # Verify IC
    max_violation = 0
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        vx_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        for t2 in range(n):
            if t == t2:
                continue
            v1_t2, v2_t2 = type_to_vals[t2]
            vx_t2 = v1_t2 * allocations[t2, 0] + v2_t2 * allocations[t2, 1]
            lhs = payments[t] - payments[t2]
            rhs = vx_t - vx_t2
            if lhs > rhs + 1e-10:
                max_violation = max(max_violation, lhs - rhs)
    
    if max_violation > 1e-9:
        # If IC is violated, fall back to safe mechanism
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
    
    return {'allocations': allocations, 'payments': payments}
