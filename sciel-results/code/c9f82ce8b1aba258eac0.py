"""Rewritten program: MODEL-AND-OPTIMIZE for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: the support of values (e.g., [0,1,2,3])
        constraints: constraint dict
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 85.0
    
    # Parse inputs
    if support is None:
        support = [0, 1, 2, 3]
    if not isinstance(support, (list, tuple)):
        support = [0, 1, 2, 3]
    support = list(support)
    n_vals = len(support)
    
    # Type space: (v1_idx, v2_idx) -> n_vals * n_vals types
    types = list(product(range(n_vals), repeat=2))
    n_types = len(types)
    
    # Determine probabilities from distribution_spec
    probs = _parse_distribution(distribution_spec, n_vals, n_types)
    
    # Values matrix
    V = np.array([[support[types[i][0]], support[types[i][1]]] for i in range(n_types)], dtype=np.float64)
    
    # Budget check
    if time.time() > deadline - 5.0:
        return _baseline_result(V, probs)
    
    from scipy.optimize import linprog
    
    # Variables: x = [a[0,0], a[0,1], ..., a[n-1,0], a[n-1,1], p[0], ..., p[n-1]]
    def idx_a(t, j):
        return 2 * t + j
    
    def idx_p(t):
        return 2 * n_types + t
    
    n_vars = 3 * n_types
    
    # Objective: maximize sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[idx_p(t)] = -probs[t]
    
    # Build constraint matrix
    # Truthfulness: for all t, tp: V[t]·a[t] - p[t] >= V[t]·a[tp] - p[tp]
    # Rewritten: -V[t]·a[t] + V[t]·a[tp] + p[t] - p[tp] <= 0
    # IR: V[t]·a[t] - p[t] >= 0
    # Rewritten: -V[t]·a[t] + p[t] <= 0
    
    n_ineq = n_types * n_types + n_types  # truthfulness + IR
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    row = 0
    for t in range(n_types):
        for tp in range(n_types):
            # Truthfulness: -V[t]·a[t] + V[t]·a[tp] + p[t] - p[tp] <= 0
            for j in range(2):
                A_ub[row, idx_a(t, j)] = -V[t, j]
                A_ub[row, idx_a(tp, j)] = V[t, j]
            A_ub[row, idx_p(t)] = 1.0
            A_ub[row, idx_p(tp)] = -1.0
            b_ub[row] = 0.0
            row += 1
    
    for t in range(n_types):
        # IR: -V[t]·a[t] + p[t] <= 0
        for j in range(2):
            A_ub[row, idx_a(t, j)] = -V[t, j]
        A_ub[row, idx_p(t)] = 1.0
        b_ub[row] = 0.0
        row += 1
    
    # Bounds
    bounds = []
    for t in range(n_types):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for t in range(n_types):
        bounds.append((0.0, None))
    
    # Check time
    if time.time() > deadline - 3.0:
        return _baseline_result(V, probs)
    
    # Solve LP
    try:
        result = linprog(
            c, 
            A_ub=A_ub, 
            b_ub=b_ub, 
            bounds=bounds, 
            method='highs',
            options={
                'presolve': True,
                'dual_feasibility_tolerance': 1e-9,
                'primal_feasibility_tolerance': 1e-9,
                'time_limit': 60.0
            }
        )
    except Exception as e:
        return _baseline_result(V, probs)
    
    if result.success:
        x = result.x
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for t in range(n_types):
            for j in range(2):
                allocations[t, j] = x[idx_a(t, j)]
            payments[t] = x[idx_p(t)]
        
        # Ensure non-negativity and bounds strictly
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)
        
        # Verify with tight tolerance
        if _verify_mechanism(V, probs, allocations, payments, tol=1e-8):
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            # If verification fails, try a simple repair: increase payments slightly
            # to ensure IR and IC. This reduces revenue but ensures validity.
            repaired_allocations, repaired_payments = _repair_mechanism(V, allocations, payments)
            if repaired_allocations is not None:
                if _verify_mechanism(V, probs, repaired_allocations, repaired_payments, tol=1e-8):
                    return {
                        'allocations': repaired_allocations,
                        'payments': repaired_payments
                    }
    
    # Fallback
    return _baseline_result(V, probs)


def _parse_distribution(distribution_spec, n_vals, n_types):
    """Parse distribution spec into probability vector."""
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=np.float64).flatten()
                if probs.shape[0] == n_types:
                    probs = np.maximum(probs, 0.0)
                    total = probs.sum()
                    if total > 1e-12:
                        return probs / total
            elif 'type' in distribution_spec:
                dtype = distribution_spec['type']
                if dtype == 'uniform' or dtype == 'product_uniform':
                    return np.ones(n_types) / n_types
                if 'params' in distribution_spec:
                    params = distribution_spec['params']
                    if isinstance(params, dict) and 'p1' in params and 'p2' in params:
                        p1 = np.array(params['p1'], dtype=np.float64)
                        p2 = np.array(params['p2'], dtype=np.float64)
                        if len(p1) == n_vals and len(p2) == n_vals:
                            probs = np.zeros(n_types)
                            for i in range(n_vals):
                                for j in range(n_vals):
                                    probs[i * n_vals + j] = p1[i] * p2[j]
                            probs = np.maximum(probs, 0.0)
                            total = probs.sum()
                            if total > 1e-12:
                                return probs / total
    return np.ones(n_types) / n_types


def _baseline_result(V, probs):
    """Return a valid baseline mechanism."""
    n_types = len(V)
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    return {
        'allocations': allocations,
        'payments': payments
    }


def _verify_mechanism(V, probs, allocations, payments, tol=1e-8):
    """Verify mechanism constraints."""
    n_types = len(V)
    
    if np.any(allocations < -tol) or np.any(allocations > 1.0 + tol):
        return False
    if np.any(payments < -tol):
        return False
    
    for t in range(n_types):
        for tp in range(n_types):
            u_t_t = np.dot(V[t], allocations[t]) - payments[t]
            u_t_tp = np.dot(V[t], allocations[tp]) - payments[tp]
            if u_t_t < u_t_tp - tol:
                return False
    
    for t in range(n_types):
        u_t_t = np.dot(V[t], allocations[t]) - payments[t]
        if u_t_t < -tol:
            return False
    
    if probs.sum() < 1.0 - tol or probs.sum() > 1.0 + tol:
        return False
    if np.any(probs < -tol):
        return False
    
    return True

def _repair_mechanism(V, allocations, payments):
    """
    Attempt to repair a mechanism that is slightly infeasible.
    Strategy: Increase payments to satisfy IC and IR.
    This is a heuristic and may reduce revenue.
    """
    n_types = len(V)
    # We need to find p' >= p such that IC and IR hold.
    # This is another LP: Minimize sum(p') subject to IC/IR.
    # But for speed, we can just iteratively increase payments where violations occur.
    
    p_new = payments.copy()
    a = allocations
    
    # Iterate a few times
    for _ in range(10):
        max_violation = 0.0
        violation_t = -1
        violation_tp = -1
        
        # Check IC
        for t in range(n_types):
            for tp in range(n_types):
                u_t_t = np.dot(V[t], a[t]) - p_new[t]
                u_t_tp = np.dot(V[t], a[tp]) - p_new[tp]
                violation = u_t_tp - u_t_t
                if violation > max_violation:
                    max_violation = violation
                    violation_t = t
                    violation_tp = tp
        
        # Check IR
        for t in range(n_types):
            u_t_t = np.dot(V[t], a[t]) - p_new[t]
            violation = -u_t_t
            if violation > max_violation:
                max_violation = violation
                violation_t = t
                violation_tp = -1 # Special marker for IR
        
        if max_violation <= 1e-9:
            return a, p_new
        
        # If IC violation: u_t_t < u_t_tp => v_t a_t - p_t < v_t a_tp - p_tp
        # To fix, increase p_tp (reduces u_t_tp) or decrease p_t (increases u_t_t, but p_t >= 0).
        # Increasing p_tp is safer for IR of t? No, IR of t depends on p_t.
        # Increasing p_tp hurts IR of tp? No, IR of tp is v_tp a_tp - p_tp.
        # So increasing p_tp might break IR of tp.
        # Better: Increase p_t? No, that makes u_t_t smaller.
        # We need u_t_t >= u_t_tp.
        # We can increase p_tp by the amount of violation.
        if violation_tp != -1:
            p_new[violation_tp] += max_violation + 1e-9
        else:
            # IR violation: v_t a_t - p_t < 0 => p_t > v_t a_t
            # We need p_t <= v_t a_t.
            # If p_t > v_t a_t, we can reduce p_t? But p_t >= 0.
            # If v_t a_t < 0, impossible. But v, a >= 0, so v_t a_t >= 0.
            # So we can set p_t = max(0, v_t a_t - epsilon).
            # However, reducing p_t might break IC for other types.
            # Let's just increase all payments? No.
            # For IR violation, we must have p_t <= v_t a_t.
            # If current p_t > v_t a_t, we can reduce p_t to v_t a_t.
            p_new[violation_t] = max(0.0, np.dot(V[violation_t], a[violation_t]))
        
        if time.time() > 85.0:
            return None, None
            
    return None, None
# EVOLVE-BLOCK-END
