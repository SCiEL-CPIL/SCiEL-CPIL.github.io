"""Incentive-compatible finite auction design solver using LP with tighter constraints."""
import time
import numpy as np
from scipy.optimize import linprog


def _parse_distribution(distribution_spec, n_types, types, type_idx, support):
    """Parse distribution spec into probability vector."""
    vals = [0, 1, 2, 3]
    probs = np.zeros(n_types)
    
    if distribution_spec is None:
        probs = np.ones(n_types) / n_types
        return probs
    
    if isinstance(distribution_spec, dict):
        for key, p in distribution_spec.items():
            if isinstance(key, (tuple, list)):
                t = tuple(key)
                if t in type_idx:
                    probs[type_idx[t]] = float(p)
            elif isinstance(key, str):
                try:
                    parts = key.replace('(', '').replace(')', '').replace(' ', '')
                    if ',' in parts:
                        parts = parts.split(',')
                    elif ' ' in parts:
                        parts = parts.split()
                    else:
                        parts = list(parts)
                    t = (int(parts[0]), int(parts[1]))
                    if t in type_idx:
                        probs[type_idx[t]] = float(p)
                except:
                    pass
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.asarray(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            probs = arr.copy()
        else:
            probs = np.ones(n_types) / n_types
    else:
        try:
            arr = np.asarray(distribution_spec, dtype=float).flatten()
            if len(arr) == n_types:
                probs = arr.copy()
            else:
                probs = np.ones(n_types) / n_types
        except:
            probs = np.ones(n_types) / n_types
    
    # Apply support mask
    if support is not None and len(support) > 0:
        support_set = set()
        for s in support:
            if isinstance(s, (tuple, list)):
                support_set.add(tuple(s))
            elif isinstance(s, str):
                try:
                    parts = s.replace('(', '').replace(')', '').replace(' ', '')
                    if ',' in parts:
                        parts = parts.split(',')
                    t = (int(parts[0]), int(parts[1]))
                    support_set.add(t)
                except:
                    pass
        if support_set:
            mask = np.zeros(n_types, dtype=bool)
            for t in support_set:
                if t in type_idx:
                    mask[type_idx[t]] = True
            probs[~mask] = 0.0
    
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.ones(n_types) / n_types
    
    return probs


def _solve_auction_lp(probs, n_types, types, type_idx, deadline):
    """Solve the LP for optimal revenue mechanism."""
    n_goods = 2
    n_alloc = n_types * n_goods
    n_pay = n_types
    n_vars = n_alloc + n_pay
    
    # Objective: maximize sum_v probs[v] * p_v => minimize -sum
    c = np.zeros(n_vars)
    for v in range(n_types):
        c[n_alloc + v] = -probs[v]
    
    constraints_list = []
    b_list = []
    
    # IC constraints: for each true type v_true and report v_rep (v_rep != v_true):
    # u(v_true, v_rep) <= u(v_true, v_true)
    # v_true . q[v_rep] - p[v_rep] <= v_true . q[v_true] - p[v_true]
    # => p[v_true] - p[v_rep] - v_true . (q[v_true] - q[v_rep]) <= 0
    for v_true in range(n_types):
        vt = types[v_true]
        for v_rep in range(n_types):
            if v_rep == v_true:
                continue
            row = np.zeros(n_vars)
            row[n_alloc + v_true] = 1.0
            row[n_alloc + v_rep] = -1.0
            row[v_true * n_goods + 0] = -vt[0]
            row[v_rep * n_goods + 0] = vt[0]
            row[v_true * n_goods + 1] = -vt[1]
            row[v_rep * n_goods + 1] = vt[1]
            constraints_list.append(row)
            b_list.append(0.0)
    
    # IR constraints: p_v <= v . q_v => p_v - v1*q_v1 - v2*q_v2 <= 0
    for v in range(n_types):
        vt = types[v]
        row = np.zeros(n_vars)
        row[n_alloc + v] = 1.0
        row[v * n_goods + 0] = -vt[0]
        row[v * n_goods + 1] = -vt[1]
        constraints_list.append(row)
        b_list.append(0.0)
    
    # Nonnegative payments: -p_v <= 0
    for v in range(n_types):
        row = np.zeros(n_vars)
        row[n_alloc + v] = -1.0
        constraints_list.append(row)
        b_list.append(0.0)
    
    A_ub = np.array(constraints_list)
    b_ub = np.array(b_list)
    
    bounds = [(0, 1)] * n_alloc + [(0, None)] * n_pay
    
    if time.time() > deadline - 3.0:
        return None
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            x = result.x
            allocations = x[:n_alloc].reshape(n_types, n_goods)
            payments = x[n_alloc:]
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)
            return allocations, payments
    except Exception:
        pass
    
    return None


def _verify_and_fix(allocations, payments, types, n_types, deadline):
    """Verify IC and IR constraints, fix if needed."""
    n_goods = 2
    
    # Check IC: regret <= 1e-8
    max_regret = 0.0
    for v_true in range(n_types):
        vt = types[v_true]
        for v_rep in range(n_types):
            u_rep = vt[0] * allocations[v_rep, 0] + vt[1] * allocations[v_rep, 1] - payments[v_rep]
            u_truth = vt[0] * allocations[v_true, 0] + vt[1] * allocations[v_true, 1] - payments[v_true]
            regret = u_rep - u_truth
            if regret > max_regret:
                max_regret = regret
    
    if max_regret > 1e-8:
        # Try to fix by adjusting payments
        # Add uniform payment adjustment
        adj = max_regret + 1e-9
        payments_new = payments + adj
        
        # Re-check IC
        max_regret2 = 0.0
        for v_true in range(n_types):
            vt = types[v_true]
            for v_rep in range(n_types):
                u_rep = vt[0] * allocations[v_rep, 0] + vt[1] * allocations[v_rep, 1] - payments_new[v_rep]
                u_truth = vt[0] * allocations[v_true, 0] + vt[1] * allocations[v_true, 1] - payments_new[v_true]
                regret = u_rep - u_truth
                if regret > max_regret2:
                    max_regret2 = regret
        
        # Check IR with new payments
        ir_ok = True
        for v in range(n_types):
            vt = types[v]
            if payments_new[v] - (vt[0] * allocations[v, 0] + vt[1] * allocations[v, 1]) > 1e-8:
                ir_ok = False
                break
        
        if max_regret2 <= 1e-8 and ir_ok:
            payments = payments_new
        else:
            # Fallback: zero mechanism
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
    
    # Final IR check
    for v in range(n_types):
        vt = types[v]
        max_pay = vt[0] * allocations[v, 0] + vt[1] * allocations[v, 1]
        if payments[v] > max_pay + 1e-8:
            payments[v] = max(0, max_pay)
    
    # Ensure nonnegative
    payments = np.clip(payments, 0, None)
    
    return allocations, payments


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    type_idx = {t: i for i, t in enumerate(types)}
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, n_types, types, type_idx, support)
    
    # Solve LP
    result = _solve_auction_lp(probs, n_types, types, type_idx, deadline)
    
    if result is not None:
        allocations, payments = result
        allocations, payments = _verify_and_fix(allocations, payments, types, n_types, deadline)
    else:
        # Fallback: trivial mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
    
    return {
        'allocations': allocations,
        'payments': payments
    }
