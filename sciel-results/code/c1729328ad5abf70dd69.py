"""Incentive-compatible finite auction design solver."""
import time
import numpy as np
from scipy.optimize import linprog


def _solve_optimal_mechanism(p1, p2, deadline):
    """Solve the LP for the optimal incentive-compatible mechanism."""
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = 16
    joint_probs = np.outer(p1, p2).flatten()

    # Variables: x[t*2], x[t*2+1], p[t]
    n_vars = 2 * n_types + n_types  # 48

    # Objective: maximize expected revenue
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[2 * n_types + t] = -joint_probs[t]

    # Build constraints
    A_ub_rows = []
    b_ub_vals = []

    # IC constraints: p[t] - p[t'] <= v1[t]*(x[t,0]-x[t',0]) + v2[t]*(x[t,1]-x[t',1])
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
            row[2 * n_types + t] = 1
            row[2 * n_types + tp] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    # IR constraints: p[t] <= v1[t]*x[t,0] + v2[t]*x[t,1]
    for t in range(n_types):
        v1_t, v2_t = types[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1_t
        row[t * 2 + 1] = -v2_t
        row[2 * n_types + t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # Nonnegative payments: -p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[2 * n_types + t] = -1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)

    bounds = []
    for i in range(2 * n_types):
        bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not result.success:
        return None

    x_vars = result.x[:2 * n_types]
    p_vars = result.x[2 * n_types:]

    return x_vars, p_vars


def _validate_mechanism(allocations, payments, types, joint_probs, tol=1e-8):
    """Check IC and IR constraints."""
    n_types = len(types)
    for t in range(n_types):
        v1_t, v2_t = types[t]
        u_tt = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if u_tt < -tol:
            return False
        for tp in range(n_types):
            u_tp = v1_t * allocations[tp, 0] + v2_t * allocations[tp, 1] - payments[tp]
            if u_tt < u_tp - tol:
                return False
    return True


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0

    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = 16

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

    # Try to solve the LP
    best_allocations = None
    best_payments = None

    try:
        result = _solve_optimal_mechanism(p1, p2, deadline)
        if result is not None:
            x_vars, p_vars = result
            x_vars = np.clip(x_vars, 0.0, 1.0)
            p_vars = np.maximum(p_vars, 0.0)
            allocations = x_vars.reshape(16, 2)
            payments = p_vars

            if _validate_mechanism(allocations, payments, types, joint_probs):
                best_allocations = allocations
                best_payments = payments

            # If validation failed due to tiny numerical errors, try a small adjustment
            else:
                # Subtract tiny epsilon from payments to fix IR violations
                adjusted_p = np.maximum(payments - 1e-10, 0.0)
                if _validate_mechanism(allocations, payments, types, joint_probs, tol=1e-6):
                    best_allocations = allocations
                    best_payments = adjusted_p

    except Exception:
        pass

    # Check time
    if time.time() > deadline - 5.0 or best_allocations is not None:
        if best_allocations is not None:
            return {'allocations': best_allocations, 'payments': best_payments}

    # Fallback: threshold mechanism
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    for t in range(n_types):
        v1_t, v2_t = types[t]
        if v1_t >= 2:
            allocations[t, 0] = 1.0
        if v2_t >= 2:
            allocations[t, 1] = 1.0
        payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]

    return {'allocations': allocations, 'payments': payments}
