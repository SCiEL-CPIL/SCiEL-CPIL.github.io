"""Incentive-compatible finite auction design solver – dual max-min bisection."""
import time
import numpy as np
from scipy.optimize import linprog


def _parse_distribution(distribution_spec):
    values = [0, 1, 2, 3]
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25]*4))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25]*4))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec); p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4]); p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25]*4; p2 = [0.25]*4
    else:
        p1 = [0.25]*4; p2 = [0.25]*4
    p1 = np.array(p1, dtype=float); p2 = np.array(p2, dtype=float)
    if p1.sum() > 0: p1 = p1 / p1.sum()
    else: p1 = np.ones(4)/4
    if p2.sum() > 0: p2 = p2 / p2.sum()
    else: p2 = np.ones(4)/4
    return p1, p2


def _solve_maxmin(p1, p2, R):
    """
    Max-min LP: maximize u such that for all t:
      u <= sum_g v_t,g * x_t,g - p_t
    with IC, IR, 0<=x<=1, p>=0, and revenue constraint:
      sum_t Pr[t]*p_t >= R
    Variables: x[0..31], p[0..15], u -> 48 vars
    But we use the dual trick: for fixed R, check feasibility of:
      exists x,p,u s.t. u <= utility_t for all t, IC, IR, rev>=R
    This is still an LP. We solve it directly.
    """
    n_types = 16
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    joint_probs = np.outer(p1, p2).flatten()

    n_x = 32
    n_p = 16
    n_u = 1
    n_vars = n_x + n_p + n_u

    # Maximize u -> minimize -u
    c = np.zeros(n_vars)
    c[n_x + n_p] = -1.0  # u variable index

    A_ub_rows = []
    b_ub_vals = []

    # Utility constraints: u - sum_g v_t,g * x_t,g + p_t <= 0  for all t
    for t in range(n_types):
        v1_t, v2_t = types[t]
        row = np.zeros(n_vars)
        row[t*2] = -v1_t
        row[t*2+1] = -v2_t
        row[n_x + t] = 1.0
        row[n_x + n_p] = 1.0  # u
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # IC: p[t] - p[tp] <= v_t . (x_t - x_tp)
    # => -v_t,0*x_t,0 - v_t,1*x_t,1 + v_t,0*x_tp,0 + v_t,1*x_tp,1 + p[t] - p[tp] <= 0
    for t in range(n_types):
        v1_t, v2_t = types[t]
        for tp in range(n_types):
            if t == tp: continue
            row = np.zeros(n_vars)
            row[t*2] = -v1_t
            row[t*2+1] = -v2_t
            row[tp*2] += v1_t
            row[tp*2+1] += v2_t
            row[n_x + t] = 1.0
            row[n_x + tp] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    # IR: p[t] <= v_t . x_t  =>  -v_t,0*x_t,0 - v_t,1*x_t,1 + p[t] <= 0
    for t in range(n_types):
        v1_t, v2_t = types[t]
        row = np.zeros(n_vars)
        row[t*2] = -v1_t
        row[t*2+1] = -v2_t
        row[n_x + t] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # Nonneg payments: -p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[n_x + t] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # Revenue: sum_t Pr[t]*p_t >= R  =>  -sum_t Pr[t]*p_t <= -R
    row = np.zeros(n_vars)
    for t in range(n_types):
        row[n_x + t] = -joint_probs[t]
    A_ub_rows.append(row)
    b_ub_vals.append(-R)

    bounds = []
    for i in range(n_x):
        bounds.append((0, 1))
    for i in range(n_p):
        bounds.append((0, None))
    bounds.append((None, None))  # u unbounded

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return None
        x_vars = res.x[:n_x]
        p_vars = res.x[n_x:n_x+n_p]
        x_vars = np.clip(x_vars, 0, 1)
        p_vars = np.maximum(p_vars, 0)
        return x_vars.reshape(16, 2), p_vars
    except Exception:
        return None


def _validate(allocations, payments, types):
    tol = 1e-8
    for t in range(16):
        v1_t, v2_t = types[t]
        u_tt = v1_t * allocations[t,0] + v2_t * allocations[t,1] - payments[t]
        if u_tt < -tol:
            return False
        for tp in range(16):
            u_tp = v1_t * allocations[tp,0] + v2_t * allocations[tp,1] - payments[tp]
            if u_tt < u_tp - tol:
                return False
    return True


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin

    p1, p2 = _parse_distribution(distribution_spec)
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    joint_probs = np.outer(p1, p2).flatten()

    # Upper bound on revenue: each type pays at most v1+v2 <= 6
    R_max = 6.0
    R_min = 0.0

    best_alloc = None
    best_pay = None
    best_R = 0.0

    # Bisection on R
    for _ in range(40):  # ~40 iterations plenty for 1e-10 precision
        if time.time() > deadline:
            break
        R_mid = (R_min + R_max) / 2.0
        sol = _solve_maxmin(p1, p2, R_mid)
        if sol is not None:
            alloc, pay = sol
            if _validate(alloc, pay, types):
                best_alloc = alloc
                best_pay = pay
                best_R = R_mid
                R_min = R_mid
            else:
                R_max = R_mid
        else:
            R_max = R_mid

    if best_alloc is not None:
        return {'allocations': best_alloc, 'payments': best_pay}

    # Fallback: threshold mechanism
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    for t in range(16):
        v1_t, v2_t = types[t]
        if v1_t >= 2: allocations[t, 0] = 1
        if v2_t >= 2: allocations[t, 1] = 1
        payments[t] = v1_t * allocations[t,0] + v2_t * allocations[t,1]
    return {'allocations': allocations, 'payments': payments}
