"""Direct convex LP over allocation and payment variables for optimal finite-support mechanism."""
import numpy as np
from scipy.optimize import linprog
import time


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
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Parse distribution
    values = [0, 1, 2, 3]
    
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Parse support
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n = 16
    assert len(types) == n, f"Expected 16 types, got {len(types)}"
    
    # Value matrix: val[t] = (v1, v2)
    val_matrix = np.array(types, dtype=float)  # shape (16, 2)
    
    # Direct LP: variables are [x1_0,...,x1_15, x2_0,...,x2_15, p_0,...,p_15]
    # 32 allocation vars + 16 payment vars = 48 total
    #
    # Maximize: sum_t probs[t] * p[t]
    #
    # Subject to:
    # 1. IC: p[t] - p[t'] <= v(t)·x(t) - v(t)·x(t') for all t, t'
    #    => p[t] - p[t'] - v1(t)*x1[t] - v2(t)*x2[t] + v1(t)*x1[t'] + v2(t)*x2[t'] <= 0
    # 2. IR: p[t] <= v(t)·x(t) for all t
    #    => p[t] - v1(t)*x1[t] - v2(t)*x2[t] <= 0
    # 3. 0 <= x1[t], x2[t] <= 1 for all t
    # 4. p[t] >= 0 for all t
    
    # Build LP in standard form: minimize c^T z s.t. A_ub z <= b_ub, bounds
    # z = [x1(0..15), x2(0..15), p(0..15)]
    
    c = np.zeros(48)
    c[32:] = -probs  # maximize revenue = minimize -revenue
    
    A_rows = []
    b_rows = []
    
    # IC constraints: for each pair (t, t'), t != t'
    # p[t] - p[t'] - v1(t)*x1[t] - v2(t)*x2[t] + v1(t)*x1[t'] + v2(t)*x2[t'] <= 0
    for t in range(n):
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(48)
            # p[t] coefficient: +1
            row[32 + t] = 1.0
            # p[t'] coefficient: -1
            row[32 + t2] = -1.0
            # x1[t] coefficient: -v1(t)
            row[t] = -val_matrix[t, 0]
            # x2[t] coefficient: -v2(t)
            row[16 + t] = -val_matrix[t, 1]
            # x1[t'] coefficient: +v1(t)
            row[t2] = val_matrix[t, 0]
            # x2[t'] coefficient: +v2(t)
            row[16 + t2] = val_matrix[t, 1]
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints: p[t] - v1(t)*x1[t] - v2(t)*x2[t] <= 0
    for t in range(n):
        row = np.zeros(48)
        row[32 + t] = 1.0
        row[t] = -val_matrix[t, 0]
        row[16 + t] = -val_matrix[t, 1]
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: x1, x2 in [0, 1], p >= 0
    bounds = [(0, 1)] * 32 + [(0, None)] * 16
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            x1 = result.x[0:16]
            x2 = result.x[16:32]
            p = result.x[32:48]
            
            # Clip to ensure feasibility
            x1 = np.clip(x1, 0, 1)
            x2 = np.clip(x2, 0, 1)
            p = np.maximum(p, 0)
            
            alloc = np.column_stack([x1, x2])
            
            # Verify IC and IR hold (within tolerance)
            max_violation = 0.0
            for t in range(n):
                for t2 in range(n):
                    if t == t2:
                        continue
                    lhs = p[t] - p[t2]
                    rhs = np.dot(val_matrix[t], alloc[t]) - np.dot(val_matrix[t], alloc[t2])
                    violation = lhs - rhs
                    if violation > max_violation:
                        max_violation = violation
                # IR check
                ir_violation = p[t] - np.dot(val_matrix[t], alloc[t])
                if ir_violation > max_violation:
                    max_violation = ir_violation
            
            if max_violation <= 1e-8:
                return {'allocations': alloc, 'payments': p}
            else:
                # Fallback: use threshold mechanism
                return _fallback_threshold(val_matrix, probs)
        else:
            return _fallback_threshold(val_matrix, probs)
    except Exception:
        return _fallback_threshold(val_matrix, probs)


def _fallback_threshold(val_matrix, probs):
    """Fallback: use threshold mechanism if LP fails."""
    n = 16
    best_revenue = -np.inf
    best_alloc = None
    best_pay = None
    
    thresholds = [0, 1, 2, 3, 4]
    
    for a in thresholds:
        for b in thresholds:
            alloc = np.zeros((16, 2))
            for t in range(16):
                v1_t, v2_t = val_matrix[t]
                alloc[t, 0] = 1.0 if v1_t >= a else 0.0
                alloc[t, 1] = 1.0 if v2_t >= b else 0.0
            
            pay = _solve_payments(alloc, val_matrix, probs)
            if pay is None:
                continue
            
            revenue = np.sum(probs * pay)
            if revenue > best_revenue:
                best_revenue = revenue
                best_alloc = alloc.copy()
                best_pay = pay.copy()
    
    if best_alloc is None:
        best_alloc = np.ones((16, 2))
        best_pay = np.zeros(16)
    
    best_pay = np.maximum(best_pay, 0)
    best_alloc = np.clip(best_alloc, 0, 1)
    
    return {'allocations': best_alloc, 'payments': best_pay}


def _solve_payments(alloc, val_matrix, probs):
    """Solve for optimal payments given allocation rule. Returns payment vector or None."""
    n = 16
    
    # LP: maximize sum_t probs[t]*p(t)
    # Subject to:
    # 1. p(t) - p(t') <= v(t)·(x(t) - x(t')) for all t != t'
    # 2. p(t) <= v(t)·x(t) for all t  (IR)
    # 3. p(t) >= 0 for all t
    
    c = -probs  # minimize -revenue
    
    A_rows = []
    b_rows = []
    
    # IC constraints: p(t) - p(t') <= v(t)·x(t) - v(t)·x(t')
    for t in range(n):
        for t2 in range(n):
            if t == t2:
                continue
            rhs = np.dot(val_matrix[t], alloc[t]) - np.dot(val_matrix[t], alloc[t2])
            row = np.zeros(n)
            row[t] = 1
            row[t2] = -1
            A_rows.append(row)
            b_rows.append(rhs)
    
    # IR constraints: p(t) <= v(t)·x(t)
    for t in range(n):
        rhs = np.dot(val_matrix[t], alloc[t])
        row = np.zeros(n)
        row[t] = 1
        A_rows.append(row)
        b_rows.append(rhs)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    bounds = [(0, None)] * n
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return result.x
        else:
            return None
    except Exception:
        return None
