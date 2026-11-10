"""Incentive-compatible finite auction design: direct LP with tighter tolerances and polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with 1 additive bidder, 2 goods.
    Values in {0,1,2,3}. Three fixed distributions.
    Returns: {'allocations': (16, 2) array of probabilities, 'payments': (16,) array}
    """
    t0 = time.time()
    T_LIMIT = 85.0

    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    N = 16
    V = np.array(types, dtype=np.float64)

    def _elapsed():
        return time.time() - t0

    def _parse_probs():
        if isinstance(distribution_spec, dict):
            if 'probs' in distribution_spec:
                arr = np.asarray(distribution_spec['probs'], dtype=np.float64)
                if arr.shape == (N,):
                    arr = np.maximum(arr, 0.0)
                    s = arr.sum()
                    return arr / s if s > 0 else np.ones(N) / N
            if 'distribution' in distribution_spec:
                d = distribution_spec['distribution']
                if d == 'uniform':
                    return np.ones(N) / N
                if d == 'geometric':
                    base = 0.7
                    arr = np.array([base**types[i][0] * base**types[i][1] for i in range(N)])
                    return arr / arr.sum()
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.asarray(distribution_spec, dtype=np.float64)
            if arr.shape == (N,):
                arr = np.maximum(arr, 0.0)
                s = arr.sum()
                if s > 0:
                    return arr / s
        return np.ones(N) / N

    probs = _parse_probs()

    if _elapsed() > T_LIMIT - 5:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    # LP: vars = [q1_0..q1_15, q2_0..q2_15, p_0..p_15]
    n_vars = 3 * N
    c_obj = np.zeros(n_vars)
    c_obj[2*N : 3*N] = -probs

    A_rows = []
    b_vals = []

    # IR: p_i <= v1_i*q1_i + v2_i*q2_i  =>  -v1_i*q1_i - v2_i*q2_i + p_i <= 0
    for i in range(N):
        row = np.zeros(n_vars)
        row[i] = -V[i, 0]
        row[N + i] = -V[i, 1]
        row[2*N + i] = 1.0
        A_rows.append(row)
        b_vals.append(0.0)

    # IC: for all i != j:  v1_i*(q1_j - q1_i) + v2_i*(q2_j - q2_i) - p_j + p_i <= 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[j] = V[i, 0]
            row[i] = -V[i, 0]
            row[N + j] = V[i, 1]
            row[N + i] = -V[i, 1]
            row[2*N + j] = -1.0
            row[2*N + i] = 1.0
            A_rows.append(row)
            b_vals.append(0.0)

    A_ub = np.array(A_rows, dtype=np.float64)
    b_ub = np.array(b_vals, dtype=np.float64)

    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, 6.0)] * N

    remaining = T_LIMIT - _elapsed() - 5.0
    if remaining < 1.0:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    try:
        res = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={
                'presolve': True,
                'dual_feasibility_tolerance': 1e-13,
                'primal_feasibility_tolerance': 1e-13,
                'time_limit': remaining
            }
        )
    except Exception:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    if not res.success or res.x is None:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    x = res.x
    q1 = np.clip(x[:N], 0.0, 1.0)
    q2 = np.clip(x[N:2*N], 0.0, 1.0)
    p = np.maximum(x[2*N:3*N], 0.0)

    v1 = V[:, 0]
    v2 = V[:, 1]

    # Fix IR violations
    util_true = v1 * q1 + v2 * q2
    ir_viol = np.max(p - util_true)
    if ir_viol > 1e-12:
        p = np.maximum(p - ir_viol, 0.0)

    # Polish: solve a small LP to maximize revenue given fixed allocations
    # This removes numerical noise from the first solve and ensures tight IC/IR
    if _elapsed() < T_LIMIT - 3:
        try:
            # Fix q1, q2; optimize p only subject to IC + IR
            n_p = N
            c_p = -probs  # maximize sum(probs_i * p_i)
            Ap_rows = []
            bp_vals = []

            # IR: p_i <= v1_i*q1_i + v2_i*q2_i
            for i in range(N):
                row = np.zeros(n_p)
                row[i] = 1.0
                Ap_rows.append(row)
                bp_vals.append(v1[i]*q1[i] + v2[i]*q2[i])

            # IC: p_i - p_j <= v1_i*(q1_i - q1_j) + v2_i*(q2_i - q2_j)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    row = np.zeros(n_p)
                    row[i] = 1.0
                    row[j] = -1.0
                    Ap_rows.append(row)
                    bp_vals.append(v1[i]*(q1[i] - q1[j]) + v2[i]*(q2[i] - q2[j]))

            Ap_ub = np.array(Ap_rows, dtype=np.float64)
            bp_ub = np.array(bp_vals, dtype=np.float64)
            bounds_p = [(0.0, 6.0)] * N

            res2 = linprog(
                c_p,
                A_ub=Ap_ub,
                b_ub=bp_ub,
                bounds=bounds_p,
                method='highs',
                options={
                    'presolve': True,
                    'dual_feasibility_tolerance': 1e-14,
                    'primal_feasibility_tolerance': 1e-14,
                    'time_limit': max(1.0, T_LIMIT - _elapsed() - 2.0)
                }
            )
            if res2.success and res2.x is not None:
                p = np.maximum(res2.x, 0.0)
        except Exception:
            pass

    # Final verification
    util_true = v1 * q1 + v2 * q2
    if np.max(p - util_true) > 1e-10:
        p = np.maximum(p - (np.max(p - util_true) + 1e-10), 0.0)

    ic_ok = True
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            lhs = p[i] - p[j]
            rhs = v1[i]*(q1[i] - q1[j]) + v2[i]*(q2[i] - q2[j])
            if lhs - rhs > 1e-8:
                ic_ok = False
                break
        if not ic_ok:
            break

    if not ic_ok:
        # Fallback: reduce payments to fix IC
        max_viol = 0.0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                lhs = p[i] - p[j]
                rhs = v1[i]*(q1[i] - q1[j]) + v2[i]*(q2[i] - q2[j])
                max_viol = max(max_viol, lhs - rhs)
        if max_viol > 0:
            p = np.maximum(p - max_viol, 0.0)
            # Re-check
            still_bad = False
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    if p[i] - p[j] - (v1[i]*(q1[i]-q1[j]) + v2[i]*(q2[i]-q2[j])) > 1e-8:
                        still_bad = True
                        break
                if still_bad:
                    break
            if still_bad:
                return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    allocations = np.column_stack((q1, q2))
    return {'allocations': allocations, 'payments': p}
# EVOLVE-BLOCK-END
