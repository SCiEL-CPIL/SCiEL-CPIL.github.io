import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    DEADLINE = start_time + 80  # Safety margin before 90s kill

    # Parse problem structure
    values = [0, 1, 2, 3]

    # Get distribution
    probs = None
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            for key in ('probabilities', 'dist', 'p'):
                if key in distribution_spec:
                    probs = np.array(distribution_spec[key], dtype=float)
                    break
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)

    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    # Determine types
    types = [(i, j) for i in values for j in values]
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]

    n_types = len(types)
    assert n_types == 16, f"Expected 16 types, got {n_types}"

    type_to_vals = {idx: (v1, v2) for idx, (v1, v2) in enumerate(types)}
    type_vals = np.array([type_to_vals[i] for i in range(16)], dtype=float)
    v1_all = type_vals[:, 0]
    v2_all = type_vals[:, 1]

    # LP Formulation
    # Variables: x1[0..15], x2[16..31], p[32..47]
    n_vars = 48

    A_rows = []
    b_rows = []

    # IC constraints: p(t) - p(t') <= v1(t)*(x1(t) - x1(t')) + v2(t)*(x2(t) - x2(t'))
    for t in range(16):
        for t2 in range(16):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            row[t] = -v1_all[t]
            row[t2] = v1_all[t]
            row[16 + t] = -v2_all[t]
            row[16 + t2] = v2_all[t]
            row[32 + t] = 1.0
            row[32 + t2] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)

    # IR constraints: p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    for t in range(16):
        row = np.zeros(n_vars)
        row[t] = -v1_all[t]
        row[16 + t] = -v2_all[t]
        row[32 + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Objective: maximize sum p(t) * probs[t]
    c = np.zeros(n_vars)
    c[32:48] = -probs

    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16

    result = None
    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'time_limit': 70
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        pass

    if result is not None and result.success:
        x = result.x
        x1 = np.clip(x[0:16], 0, 1)
        x2 = np.clip(x[16:32], 0, 1)
        p = np.maximum(x[32:48], 0)
        allocations = np.column_stack((x1, x2))
        payments = p
    else:
        # Fallback: Zero mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)

    # Verification and Minimal Fix
    max_violation = 0.0
    for t in range(16):
        v1_t, v2_t = type_to_vals[t]
        util_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if util_t < 0:
            max_violation = max(max_violation, -util_t)

        for t2 in range(16):
            if t == t2:
                continue
            lhs = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            rhs = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
            if lhs < rhs:
                violation = rhs - lhs
                if violation > max_violation:
                    max_violation = violation

    if max_violation > 1e-8:
        # Reduce payments to fix IC/IR
        payments = np.maximum(payments - max_violation, 0.0)

        # Re-verify IR after reduction
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            util = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            if util < 0:
                payments[t] = max(0.0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])

    return {'allocations': allocations, 'payments': payments}
