import time
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    T_LIMIT = 85.0
    types = [(a, b) for a in range(4) for b in range(4)]
    N = 16
    V = np.array([[types[i][0], types[i][1]] for i in range(N)], dtype=np.float64)

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
    if _elapsed() > T_LIMIT:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    # LP variables: [x_0..x_15, y_0..y_15, p_0..p_15]
    # Revenue = sum_i probs[i] * p[i], so minimize -sum_i probs[i]*p[i]
    c_obj = np.zeros(3 * N)
    c_obj[2 * N:3 * N] = -probs  # minimize negative revenue

    # IR: for each i, p[i] <= x[i]*v1_i + y[i]*v2_i
    # => -v1_i*x[i] - v2_i*y[i] + p[i] <= 0
    A_ir = np.zeros((N, 3 * N))
    for i in range(N):
        A_ir[i, i] = -V[i, 0]
        A_ir[i, N + i] = -V[i, 1]
        A_ir[i, 2 * N + i] = 1.0
    b_ir = np.zeros(N)

    # IC: for each (i,j), i!=j:
    # (x[j]-x[i])*v1_i + (y[j]-y[i])*v2_i - p[j] + p[i] <= 0
    # Build as matrix
    n_ic = N * (N - 1)
    A_ic = np.zeros((n_ic, 3 * N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            A_ic[idx, j] = V[i, 0]
            A_ic[idx, i] = -V[i, 0]
            A_ic[idx, N + j] = V[i, 1]
            A_ic[idx, N + i] = -V[i, 1]
            A_ic[idx, 2 * N + j] = -1.0
            A_ic[idx, 2 * N + i] = 1.0
            idx += 1
    b_ic = np.zeros(n_ic)

    A_ub = np.vstack([A_ir, A_ic])
    b_ub = np.concatenate([b_ir, b_ic])

    # Payments can be at most the value of the allocation for IR
    max_payment = np.array([V[i, 0] + V[i, 1] for i in range(N)])
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, max_payment[i]) for i in range(N)]

    if _elapsed() > T_LIMIT:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
    except Exception:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    if not res.success or res.x is None:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    x = np.clip(res.x[:N], 0.0, 1.0)
    y = np.clip(res.x[N:2 * N], 0.0, 1.0)
    p = np.maximum(res.x[2 * N:3 * N], 0.0)

    # Post-process: enforce IR by capping payments
    v1 = V[:, 0]
    v2 = V[:, 1]
    p_cap = x * v1 + y * v2
    p = np.minimum(p, p_cap)
    p = np.maximum(p, 0.0)

    # Local refinement: try to increase payments while maintaining IC and IR
    # The LP should be optimal, but small numerical issues might leave revenue on the table
    if _elapsed() < T_LIMIT - 5.0:
        for _round in range(3):
            improved = False
            for i in range(N):
                # Maximum p[i] such that IR holds: p[i] <= x[i]*v1[i] + y[i]*v2[i]
                max_p_i = x[i] * V[i, 0] + y[i] * V[i, 1]
                # Check IC constraints involving p[i]
                # For type i reporting as j: util_rep - util_true <= 0
                # x[j]*V[i,0] + y[j]*V[i,1] - p[j] - (x[i]*V[i,0] + y[i]*V[i,1] - p[i]) <= 0
                # => p[i] <= p[j] + x[i]*V[i,0] + y[i]*V[i,1] - x[j]*V[i,0] - y[j]*V[i,1]
                # For type j reporting as i: util_rep - util_true <= 0
                # x[i]*V[j,0] + y[i]*V[j,1] - p[i] - (x[j]*V[j,0] + y[j]*V[j,1] - p[j]) <= 0
                # => p[i] >= p[j] + x[i]*V[j,0] + y[i]*V[j,1] - x[j]*V[j,0] - y[j]*V[j,1]
                # We want to maximize p[i], so the binding constraint is from the first form
                new_max_p_i = max_p_i
                for j in range(N):
                    if i == j:
                        continue
                    # Constraint: p[i] <= p[j] + x[i]*V[i,0] + y[i]*V[i,1] - x[j]*V[i,0] - y[j]*V[i,1]
                    upper_j = p[j] + x[i] * V[i, 0] + y[i] * V[i, 1] - x[j] * V[i, 0] - y[j] * V[i, 1]
                    new_max_p_i = min(new_max_p_i, upper_j)
                
                new_p_i = max(0.0, min(new_max_p_i, max_p_i))
                if new_p_i > p[i] + 1e-12:
                    p[i] = new_p_i
                    improved = True
            
            if not improved:
                break

    # Final verification
    ic_ok = True
    for i in range(N):
        if x[i] * V[i, 0] + y[i] * V[i, 1] - p[i] < -1e-8:
            ic_ok = False
            break
        for j in range(N):
            if i == j:
                continue
            ur = x[j] * V[i, 0] + y[j] * V[i, 1] - p[j]
            ut = x[i] * V[i, 0] + y[i] * V[i, 1] - p[i]
            if ur - ut > 1e-8:
                ic_ok = False
                break
        if not ic_ok:
            break

    if not ic_ok:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    return {'allocations': np.column_stack((x, y)), 'payments': p}
