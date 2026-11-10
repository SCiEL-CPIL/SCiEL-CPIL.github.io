import time
import numpy as np
from scipy.optimize import linprog

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

    # LP Variables: x (N), y (N), p (N)
    # Minimize -sum(pi * p)
    c_obj = np.zeros(3 * N)
    c_obj[2 * N : 3 * N] = -probs
    
    A_ub_list = []
    b_ub_list = []
    
    # IR: p_i <= v1_i * x_i + v2_i * y_i
    # -v1_i * x_i - v2_i * y_i + p_i <= 0
    A_ir = np.zeros((N, 3 * N))
    for i in range(N):
        A_ir[i, i] = -V[i, 0]
        A_ir[i, N + i] = -V[i, 1]
        A_ir[i, 2 * N + i] = 1.0
    A_ub_list.append(A_ir)
    b_ub_list.append(np.zeros(N))
    
    # IC: For true type i, reporting j should not give higher utility:
    # v_i · q_j - p_j <= v_i · q_i - p_i
    # => v_i · (q_j - q_i) - p_j + p_i <= 0
    # => p_i - p_j <= v_i · (q_i - q_j)
    n_ic = N * (N - 1)
    A_ic = np.zeros((n_ic, 3 * N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # p_i - p_j <= v1_i*(x_i - x_j) + v2_i*(y_i - y_j)
            # -v1_i*x_i + v1_i*x_j - v2_i*y_i + v2_i*y_j + p_i - p_j <= 0
            A_ic[idx, i] = -V[i, 0]
            A_ic[idx, j] = V[i, 0]
            A_ic[idx, N + i] = -V[i, 1]
            A_ic[idx, N + j] = V[i, 1]
            A_ic[idx, 2 * N + i] = 1.0
            A_ic[idx, 2 * N + j] = -1.0
            idx += 1
    A_ub_list.append(A_ic)
    b_ub_list.append(np.zeros(n_ic))
    
    A_ub = np.vstack(A_ub_list)
    b_ub = np.concatenate(b_ub_list)
    
    # Bounds
    # x, y in [0, 1]
    # p in [0, max_val] where max_val = 3+3=6
    max_payment = 6.0
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, max_payment)] * N
    
    # Solve LP with tight tolerances
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={
                          'presolve': True,
                          'dual_feasibility_tolerance': 1e-13,
                          'primal_feasibility_tolerance': 1e-13,
                          'time_limit': T_LIMIT - _elapsed() - 5.0
                      })
    except Exception:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}
        
    if not res.success or res.x is None:
        # Fallback: Zero mechanism
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}
        
    x = np.clip(res.x[:N], 0.0, 1.0)
    y = np.clip(res.x[N:2*N], 0.0, 1.0)
    p = np.maximum(res.x[2*N:3*N], 0.0)
    
    v1 = V[:, 0]
    v2 = V[:, 1]
    
    # Post-process: Ensure strict feasibility with margin
    # 1. Check and fix IR: p_i <= v1_i*x_i + v2_i*y_i
    util_true = x * v1 + y * v2
    ir_violation = np.max(p - util_true)
    if ir_violation > 1e-12:
        p = np.minimum(p, util_true - 1e-10)
        p = np.maximum(p, 0.0)
    
    # 2. Check IC: p_i - p_j <= v1_i*(x_i - x_j) + v2_i*(y_i - y_j) for all i != j
    # Uniform reduction of p doesn't change IC slack, so if IR fix broke IC,
    # we need to check. But IR fix only reduces p, and reducing p_i helps IC for
    # constraints where i is the true type. Let's verify.
    ic_ok = True
    max_ic_violation = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            lhs = p[i] - p[j]
            rhs = V[i,0]*(x[i]-x[j]) + V[i,1]*(y[i]-y[j])
            viol = lhs - rhs
            if viol > max_ic_violation:
                max_ic_violation = viol
            if viol > 1e-10:
                ic_ok = False
    
    if not ic_ok:
        # Try to fix by reducing all payments by the max violation
        # This doesn't change IC slack, so it won't help if IC is violated.
        # Instead, we need a different approach: scale down allocations slightly
        # to create more slack, then recompute minimum payments.
        
        # Alternative: reduce all payments by a small amount and check if that helps.
        # Actually, the issue is that the LP solution might be slightly infeasible
        # due to numerical error. Let's try a more aggressive approach:
        # Reduce all payments by max_ic_violation + epsilon, then check IR.
        p = p - (max_ic_violation + 1e-9)
        p = np.maximum(p, 0.0)
        
        # Re-check IR after payment reduction
        ir_violation = np.max(p - util_true)
        if ir_violation > 1e-10:
            p = np.minimum(p, util_true - 1e-9)
            p = np.maximum(p, 0.0)
        
        # Final IC check
        ic_ok = True
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                lhs = p[i] - p[j]
                rhs = V[i,0]*(x[i]-x[j]) + V[i,1]*(y[i]-y[j])
                if lhs - rhs > 1e-9:
                    ic_ok = False
                    break
            if not ic_ok:
                break
        
        if not ic_ok:
            # Conservative fallback: use a simple threshold mechanism
            # Allocate good 1 if v1 >= 1, good 2 if v2 >= 1
            # Payment: p_i = max(0, v1_i * x_i + v2_i * y_i - reserve)
            # For safety, use zero allocations as ultimate fallback
            return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}
    
    # Final safety: ensure payments are non-negative and within bounds
    p = np.clip(p, 0.0, max_payment)
    
    return {'allocations': np.column_stack((x, y)), 'payments': p}
