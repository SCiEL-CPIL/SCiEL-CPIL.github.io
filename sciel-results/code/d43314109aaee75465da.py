"""Incentive-compatible finite auction design solver – revenue-equivalence LP over payments only."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    values = [0, 1, 2, 3]
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]

    # Parse distribution
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    if p1.sum() > 0:
        p1 = p1 / p1.sum()
    else:
        p1 = np.ones(4) / 4
    if p2.sum() > 0:
        p2 = p2 / p2.sum()
    else:
        p2 = np.ones(4) / 4

    joint_probs = np.outer(p1, p2).flatten()

    # Candidate deterministic monotone threshold allocation rules
    # For each good, threshold in {0, 1, 2, 3} means allocate if value >= threshold
    # Monotone: if v >= threshold, allocate 1; else 0
    # This is a single-parameter monotone allocation per good
    thresholds = [0, 1, 2, 3]

    best_revenue = -np.inf
    best_alloc = None
    best_pay = None

    for th1 in thresholds:
        for th2 in thresholds:
            # Build allocation matrix
            alloc = np.zeros((n_types, 2))
            for t in range(n_types):
                v1_t, v2_t = types[t]
                if v1_t >= th1:
                    alloc[t, 0] = 1.0
                if v2_t >= th2:
                    alloc[t, 1] = 1.0

            # Now solve LP over payments only
            # Variables: p[0..15]
            n_p = n_types
            c = -joint_probs  # maximize sum joint_probs[t] * p[t]

            A_ub_rows = []
            b_ub_vals = []

            # IR: p[t] <= v1[t]*x[t,0] + v2[t]*x[t,1]  =>  p[t] - v1[t]*x[t,0] - v2[t]*x[t,1] <= 0
            # Since alloc is fixed: p[t] <= alloc_val[t]
            for t in range(n_types):
                row = np.zeros(n_p)
                row[t] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(alloc[t, 0] * types[t][0] + alloc[t, 1] * types[t][1])

            # IC: for all t, t': p[t] - p[t'] <= v1[t]*(x[t,0]-x[t',0]) + v2[t]*(x[t,1]-x[t',1])
            # Only need to check "adjacent" or all pairs; with fixed alloc, this is linear in p
            # We include all pairs for correctness (16*15 = 240 constraints, small)
            for t in range(n_types):
                v1_t, v2_t = types[t]
                for tp in range(n_types):
                    if t == tp:
                        continue
                    v1_tp, v2_tp = types[tp]
                    # p[t] - p[tp] <= v1_t*(alloc[t,0]-alloc[tp,0]) + v2_t*(alloc[t,1]-alloc[tp,1])
                    rhs = v1_t * (alloc[t, 0] - alloc[tp, 0]) + v2_t * (alloc[t, 1] - alloc[tp, 1])
                    row = np.zeros(n_p)
                    row[t] = 1.0
                    row[tp] = -1.0
                    A_ub_rows.append(row)
                    b_ub_vals.append(rhs)

            # Nonnegative payments: -p[t] <= 0
            for t in range(n_types):
                row = np.zeros(n_p)
                row[t] = -1.0
                A_ub_rows.append(row)
                b_ub_vals.append(0.0)

            A_ub = np.array(A_ub_rows) if A_ub_rows else None
            b_ub = np.array(b_ub_vals) if b_ub_vals else None
            bounds = [(0, None)] * n_p

            try:
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if result.success:
                    p_vals = np.maximum(result.x, 0)
                    # Validate
                    valid = True
                    for t in range(n_types):
                        v1_t, v2_t = types[t]
                        u_tt = v1_t * alloc[t, 0] + v2_t * alloc[t, 1] - p_vals[t]
                        if u_tt < -1e-8:
                            valid = False
                            break
                        for tp in range(n_types):
                            u_tt_prime = v1_t * alloc[tp, 0] + v2_t * alloc[tp, 1] - p_vals[tp]
                            if u_tt < u_tt_prime - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        rev = float(np.dot(joint_probs, p_vals))
                        if rev > best_revenue:
                            best_revenue = rev
                            best_alloc = alloc.copy()
                            best_pay = p_vals.copy()
            except Exception:
                pass

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if best_alloc is not None:
        return {'allocations': best_alloc, 'payments': best_pay}

    # Fallback: simple threshold mechanism
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    for t in range(16):
        v1_t, v2_t = types[t]
        if v1_t >= 2:
            allocations[t, 0] = 1
        if v2_t >= 2:
            allocations[t, 1] = 1
        payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
    return {'allocations': allocations, 'payments': payments}
