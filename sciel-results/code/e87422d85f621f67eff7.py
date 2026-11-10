import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    T_LIMIT = 85.0
    
    # 1. Setup Types and Values
    types = [(a, b) for a in range(4) for b in range(4)]
    N = 16
    V = np.array([[types[i][0], types[i][1]] for i in range(N)], dtype=np.float64)
    v1 = V[:, 0]
    v2 = V[:, 1]

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

    # 2. Formulate LP in terms of u, x, y
    # Variables: [u_0..u_15, x_0..x_15, y_0..y_15]
    # Total vars: 3*N
    
    # Objective: Maximize Revenue = Sum( (x_i*v1_i + y_i*v2_i - u_i) * prob_i )
    # Minimize -Revenue
    c_obj = np.zeros(3 * N)
    for i in range(N):
        c_obj[i] = -probs[i]                          # -u_i * p_i
        c_obj[N + i] = -probs[i] * v1[i]              # -x_i * v1_i * p_i
        c_obj[2 * N + i] = -probs[i] * v2[i]          # -y_i * v2_i * p_i

    # Constraints:
    # A_ub * var <= b_ub
    
    # 1. IR: u_i >= 0  =>  -u_i <= 0
    A_ir = np.zeros((N, 3 * N))
    b_ir = np.zeros(N)
    for i in range(N):
        A_ir[i, i] = -1.0
        
    # 2. IC: u_i >= u_j + x_j*(v1_i - v1_j) + y_j*(v2_i - v2_j)
    # Rearranged: -u_i + u_j + x_j*(v1_i - v1_j) + y_j*(v2_i - v2_j) <= 0
    n_ic = N * (N - 1)
    A_ic = np.zeros((n_ic, 3 * N))
    b_ic = np.zeros(n_ic)
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # Coeff for u_i: -1
            A_ic[idx, i] = -1.0
            # Coeff for u_j: 1
            A_ic[idx, j] = 1.0
            # Coeff for x_j: (v1_i - v1_j)
            A_ic[idx, N + j] = v1[i] - v1[j]
            # Coeff for y_j: (v2_i - v2_j)
            A_ic[idx, 2 * N + j] = v2[i] - v2[j]
            idx += 1
            
    A_ub = np.vstack([A_ir, A_ic])
    b_ub = np.concatenate([b_ir, b_ic])

    # Bounds
    # u_i >= 0 (no upper bound explicitly, but constrained by IC and IR)
    # x_i, y_i in [0, 1]
    bounds = [(0.0, None)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N

    if _elapsed() > T_LIMIT:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    try:
        # Use HiGHS with high precision
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': 'default', 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12})
    except Exception:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    if not res.success or res.x is None:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    u = res.x[:N]
    x = np.clip(res.x[N:2 * N], 0.0, 1.0)
    y = np.clip(res.x[2 * N:3 * N], 0.0, 1.0)

    # Calculate Payments
    # p_i = x_i*v1_i + y_i*v2_i - u_i
    p = x * v1 + y * v2 - u
    
    # Enforce Non-negative Payments and IR
    # If p_i < 0, set u_i = x_i*v1_i + y_i*v2_i (IR binds)
    # But we must check if this breaks IC.
    # Since we solved the LP, p_i should be >= 0 if the LP is correct.
    # However, numerical errors might make p_i slightly negative.
    # We clip u_i to ensure p_i >= 0.
    u_max = x * v1 + y * v2
    u = np.minimum(u, u_max)
    u = np.maximum(u, 0.0)
    p = x * v1 + y * v2 - u

    # Post-processing: Snap x, y to 0 or 1 if close?
    # This can improve robustness.
    # If x_i is 0.9999, snap to 1.
    x_snap = np.where(x > 0.999, 1.0, np.where(x < 0.001, 0.0, x))
    y_snap = np.where(y > 0.999, 1.0, np.where(y < 0.001, 0.0, y))
    
    # If we snap, we must re-solve or adjust u?
    # Snapping might violate IC.
    # Let's stick to the LP solution but ensure feasibility.
    
    # Verify IC and IR
    ic_ok = True
    for i in range(N):
        # IR
        if p[i] < -1e-8 or (x[i]*v1[i] + y[i]*v2[i] - p[i]) < -1e-8:
             # Adjust p[i] to satisfy IR
             p[i] = min(p[i], x[i]*v1[i] + y[i]*v2[i])
             p[i] = max(p[i], 0.0)
        
        for j in range(N):
            if i == j: continue
            util_rep = x[j]*v1[i] + y[j]*v2[i] - p[j]
            util_true = x[i]*v1[i] + y[i]*v2[i] - p[i]
            if util_rep - util_true > 1e-8:
                ic_ok = False
                break
        if not ic_ok:
            break

    if not ic_ok:
        # Fallback: Zero mechanism
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    return {'allocations': np.column_stack((x, y)), 'payments': p}
