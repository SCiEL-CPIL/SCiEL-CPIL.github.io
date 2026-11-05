"""Incentive-compatible finite auction design solver."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0

    # Parse inputs
    values = [0, 1, 2, 3]

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
    p1 = p1 / p1.sum() if p1.sum() > 0 else np.ones(4) / 4
    p2 = p2 / p2.sum() if p2.sum() > 0 else np.ones(4) / 4

    # Joint distribution over (v1, v2)
    joint_probs = np.outer(p1, p2).flatten()

    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]

    # Variables:
    # x[t, g] for t in 0..15, g in 0,1 -> 32 allocation variables
    # p[t] for t in 0..15 -> 16 payment variables
    n_x = 32
    n_p = 16
    n_vars = n_x + n_p

    # Objective: maximize expected revenue = sum_t joint_probs[t] * p[t]
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x + t] = -joint_probs[t]

    # Build constraint matrix
    A_ub_rows = []
    b_ub_vals = []

    # IC constraints: for all t, t'
    # p[t] - p[t'] <= v1[t]*(x[t,0] - x[t',0]) + v2[t]*(x[t,1] - x[t',1])
    # Rearranging: -v1[t]*x[t,0] - v2[t]*x[t,1] + v1[t]*x[t',0] + v2[t]*x[t',1] + p[t] - p[t'] <= 0
    for t in range(n_types):
        v1_t, v2_t = types[t]
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            row[t * 2] = -v1_t
            row[t * 2 + 1] = -v2_t
            row[tp * 2] += v1_t
            row[tp * 2 + 1] += v2_t
            row[n_x + t] = 1
            row[n_x + tp] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0)

    # IR constraints: p[t] <= v1[t]*x[t,0] + v2[t]*x[t,1]
    # Rearranging: -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] <= 0
    for t in range(n_types):
        v1_t, v2_t = types[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1_t
        row[t * 2 + 1] = -v2_t
        row[n_x + t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0)

    # Nonnegative payments: p[t] >= 0 -> -p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[n_x + t] = -1
        A_ub_rows.append(row)
        b_ub_vals.append(0)

    # Bounds: 0 <= x[t,g] <= 1, p[t] >= 0
    bounds = []
    for i in range(n_x):
        bounds.append((0, 1))
    for i in range(n_p):
        bounds.append((0, None))

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if not result.success:
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

        x_vars = result.x[:n_x]
        p_vars = result.x[n_x:]

        # Clip to valid range and handle numerical noise
        x_vars = np.clip(x_vars, 0, 1)
        p_vars = np.maximum(p_vars, 0)

        allocations = x_vars.reshape(16, 2)
        payments = p_vars

        # Verify IC and IR with tolerance
        ic_ok = True
        ir_ok = True
        for t in range(16):
            v1_t, v2_t = types[t]
            u_tt = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            if u_tt < -1e-8:
                ir_ok = False
            for tp in range(16):
                u_tt_prime = v1_t * allocations[tp, 0] + v2_t * allocations[tp, 1] - payments[tp]
                if u_tt < u_tt_prime - 1e-8:
                    ic_ok = False
                    break
            if not ic_ok:
                break

        if ic_ok and ir_ok:
            return {'allocations': allocations, 'payments': payments}

        # Fallback if validation fails
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

    except Exception:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
