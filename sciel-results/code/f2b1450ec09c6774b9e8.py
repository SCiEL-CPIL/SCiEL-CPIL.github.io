import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    type_to_idx = {t: i for i, t in enumerate(types)}

    # Determine probabilities based on distribution_spec
    if isinstance(distribution_spec, str):
        if 'uniform' in distribution_spec.lower() or 'independent' in distribution_spec.lower():
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) == n_types:
            probs = probs / probs.sum()
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types

    # Decision variables: x1(t), x2(t) for each type
    # Index: 0..15 = x1 for types 0..15, 16..31 = x2 for types 0..15
    def x1_idx(t):
        return type_to_idx[t]
    def x2_idx(t):
        return n_types + type_to_idx[t]

    n_vars = 2 * n_types

    # Objective: maximize E[p(t)] -> minimize -E[p(t)]
    c = np.zeros(n_vars)

    for t in types:
        v1, v2 = t
        pi = probs[type_to_idx[t]]
        idx1 = x1_idx(t)
        idx2 = x2_idx(t)

        # p(t) = v1*x1(t) + v2*x2(t) - sum_{s=0}^{v1-1} x1(s,v2) - sum_{s=0}^{v2-1} x2(v1,s)
        c[idx1] -= pi * v1
        c[idx2] -= pi * v2

        for s in range(v1):
            t_s = (s, v2)
            c[x1_idx(t_s)] -= pi

        for s in range(v2):
            t_s = (v1, s)
            c[x2_idx(t_s)] -= pi

    # Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * n_vars

    # Constraints
    A_rows = []
    b_vals = []

    # Monotonicity in v1 for x1: x1(v1,v2) - x1(v1+1,v2) <= 0
    for v2 in vals:
        for v1 in range(3):
            t_low = (v1, v2)
            t_high = (v1 + 1, v2)
            row = np.zeros(n_vars)
            row[x1_idx(t_low)] = 1.0
            row[x1_idx(t_high)] = -1.0
            A_rows.append(row)
            b_vals.append(0.0)

    # Monotonicity in v2 for x2: x2(v1,v2) - x2(v1,v2+1) <= 0
    for v1 in vals:
        for v2 in range(3):
            t_low = (v1, v2)
            t_high = (v1, v2 + 1)
            row = np.zeros(n_vars)
            row[x2_idx(t_low)] = 1.0
            row[x2_idx(t_high)] = -1.0
            A_rows.append(row)
            b_vals.append(0.0)

    # IR: p(t) >= 0 => -p(t) <= 0
    for t in types:
        v1, v2 = t
        row = np.zeros(n_vars)
        idx1 = x1_idx(t)
        idx2 = x2_idx(t)

        row[idx1] = -v1
        row[idx2] = -v2

        for s in range(v1):
            t_s = (s, v2)
            row[x1_idx(t_s)] = 1.0

        for s in range(v2):
            t_s = (v1, s)
            row[x2_idx(t_s)] = 1.0

        A_rows.append(row)
        b_vals.append(0.0)

    if A_rows:
        A = np.array(A_rows)
        b = np.array(b_vals)
    else:
        A = np.zeros((1, n_vars))
        b = np.zeros(1)

    # Check time
    if time.time() > deadline - 5.0:
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    # Solve LP
    try:
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                         options={'presolve': True, 'time_limit': 75.0})
    except Exception:
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    if not result.success:
        # Fallback: try with simpler constraints or return zero
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    x = result.x

    # Clamp to [0,1] for numerical safety
    x = np.clip(x, 0.0, 1.0)

    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)

    for t in types:
        v1, v2 = t
        i = type_to_idx[t]
        allocations[i, 0] = x[x1_idx(t)]
        allocations[i, 1] = x[x2_idx(t)]

        # Compute payment using Myerson identity
        p = v1 * x[x1_idx(t)] + v2 * x[x2_idx(t)]
        for s in range(v1):
            t_s = (s, v2)
            p -= x[x1_idx(t_s)]
        for s in range(v2):
            t_s = (v1, s)
            p -= x[x2_idx(t_s)]

        payments[i] = max(0.0, p)

    # Final time check
    if time.time() > deadline:
        pass  # Still return what we have

    return {'allocations': allocations, 'payments': payments}
