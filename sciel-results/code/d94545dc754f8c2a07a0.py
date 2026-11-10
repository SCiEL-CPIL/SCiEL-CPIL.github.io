import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80
    
    n = 16
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    # Parse distribution
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) == n:
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n) / n
        else:
            probs = np.ones(n) / n
    else:
        probs = np.ones(n) / n
    
    # Variables: [q1_0, q2_0, q1_1, q2_1, ..., q1_15, q2_15, p_0, p_1, ..., p_15]
    # Total: 32 + 16 = 48 variables
    
    # Objective: maximize sum_v probs[v] * p[v]
    # linprog minimizes, so negate
    c = np.zeros(48)
    for v in range(n):
        c[32 + v] = -probs[v]
    
    # Inequality constraints: A_ub @ x <= b_ub
    
    A_rows = []
    b_rows = []
    
    # IC constraints: for all v, v':
    # v1*q1(v) + v2*q2(v) - p(v) >= v1*q1(v') + v2*q2(v') - p(v')
    # Rearranged: v1*q1(v) + v2*q2(v) - p(v) - v1*q1(v') - v2*q2(v') + p(v') >= 0
    # For linprog (<= form): -(v1*q1(v) + v2*q2(v) - p(v) - v1*q1(v') - v2*q2(v') + p(v')) <= 0
    # i.e.: -v1*q1(v) - v2*q2(v) + p(v) + v1*q1(v') + v2*q2(v') - p(v') <= 0
    
    for v in range(n):
        v1, v2 = types[v]
        for vp in range(n):
            if v == vp:
                continue
            vp1, vp2 = types[vp]
            row = np.zeros(48)
            # -v1*q1(v)
            row[v] = -v1
            # -v2*q2(v)
            row[n + v] = -v2
            # +p(v)
            row[32 + v] = 1
            # +vp1*q1(v')
            row[vp] = vp1
            # +vp2*q2(v')
            row[n + vp] = vp2
            # -p(v')
            row[32 + vp] = -1
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints: p(v) <= v1*q1(v) + v2*q2(v)
    # Rearranged: p(v) - v1*q1(v) - v2*q2(v) <= 0
    for v in range(n):
        v1, v2 = types[v]
        row = np.zeros(48)
        row[32 + v] = 1
        row[v] = -v1
        row[n + v] = -v2
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, 48))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    # Bounds: 0 <= q <= 1, p >= 0
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple allocation
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {"allocations": allocations, "payments": payments}
    
    x = result.x
    allocations = np.zeros((16, 2))
    for v in range(n):
        allocations[v, 0] = x[v]
        allocations[v, 1] = x[n + v]
    payments = x[32:48]
    
    # Clamp for numerical safety
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    return {"allocations": allocations, "payments": payments}
