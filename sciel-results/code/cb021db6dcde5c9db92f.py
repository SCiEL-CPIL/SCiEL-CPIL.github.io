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

    # LP variables: [x_0..x_15, y_0..y_15, p_0..p_15]
    # Revenue = sum_i probs[i] * p[i], so minimize -sum_i probs[i]*p[i]
    c_obj = np.zeros(3 * N)
    c_obj[2 * N:3 * N] = -probs  # minimize negative revenue
    
    # Precompute IC constraint matrix more efficiently
    # For each type i, the IC constraint against reporting j is:
    # v_i . (a_j - a_i) - (p_j - p_i) <= 0
    # => v_i[0]*(x_j - x_i) + v_i[1]*(y_j - y_i) - p_j + p_i <= 0
    # => -v_i[0]*x_i + v_i[0]*x_j - v_i[1]*y_i + v_i[1]*y_j + p_i - p_j <= 0
    # Note: we can use the fact that v_i = (V[i,0], V[i,1])
    # And for the allocation, a_i = (x_i, y_i)
    # So v_i . a_j - v_i . a_i - p_j + p_i <= 0
    # => (v_i . a_j - p_j) - (v_i . a_i - p_i) <= 0
    # => u_i(j) - u_i(i) <= 0
    # where u_i(j) = v_i . a_j - p_j is the utility of type i when reporting j

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
    # Build as matrix - optimized construction
    n_ic = N * (N - 1)
    A_ic = np.zeros((n_ic, 3 * N))
    idx = 0
    for i in range(N):
        vi0 = V[i, 0]
        vi1 = V[i, 1]
        for j in range(N):
            if i == j:
                continue
            A_ic[idx, j] = vi0
            A_ic[idx, i] = -vi0
            A_ic[idx, N + j] = vi1
            A_ic[idx, N + i] = -vi1
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

    # Refine Payments with Secondary LP
    # Fix x, y. Optimize p.
    # Objective: Max sum(probs * p) -> Min -sum(probs * p)
    c_p = -probs
    
    # Precompute values for efficiency
    val_x = x * V[:, 0]
    val_y = y * V[:, 1]
    total_val = val_x + val_y
    
    # Constraints for p:
    # IR: p_i <= V_i . a_i  =>  p_i - V_i . a_i <= 0
    A_p_ir = np.eye(N)
    b_p_ir = total_val.copy()
    
    # IC: p_i - p_j <= V_i . (a_i - a_j)
    # => p_i - p_j <= val_i - val_j where val_i = V_i . a_i
    # But wait, the IC constraint is:
    # u_i(i) >= u_i(j) for all i,j
    # V_i . a_i - p_i >= V_i . a_j - p_j
    # p_i - p_j <= V_i . a_i - V_i . a_j
    # Note: V_i . a_i = x[i]*V[i,0] + y[i]*V[i,1]
    # But V_i . a_j = x[j]*V[i,0] + y[j]*V[i,1]  (NOT the same as val_j!)
    A_p_ic = np.zeros((n_ic, N))
    b_p_ic = np.zeros(n_ic)
    idx = 0
    for i in range(N):
        vi0 = V[i, 0]
        vi1 = V[i, 1]
        for j in range(N):
            if i == j:
                continue
            A_p_ic[idx, i] = 1.0
            A_p_ic[idx, j] = -1.0
            # RHS: V_i . a_i - V_i . a_j = (x[i]*vi0 + y[i]*vi1) - (x[j]*vi0 + y[j]*vi1)
            b_p_ic[idx] = (x[i] * vi0 + y[i] * vi1) - (x[j] * vi0 + y[j] * vi1)
            idx += 1
            
    A_ub_p = np.vstack([A_p_ir, A_p_ic])
    b_ub_p = np.concatenate([b_p_ir, b_p_ic])
    
    bounds_p = [(0.0, b_p_ir[i]) for i in range(N)]
    
    if _elapsed() < T_LIMIT - 5.0:
        try:
            res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs',
                            options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                     'primal_feasibility_tolerance': 1e-12})
            if res_p.success and res_p.x is not None:
                p_refined = np.maximum(res_p.x, 0.0)
                # Ensure IR is strictly satisfied (numerical safety)
                p_cap = total_val
                p_refined = np.minimum(p_refined, p_cap)
                p = p_refined
        except Exception:
            pass # Keep original p

    # Final verification with tighter tolerance check
    ic_ok = True
    for i in range(N):
        # Check IR
        if total_val[i] - p[i] < -1e-7:
            ic_ok = False
            break
        # Check IC
        for j in range(N):
            if i == j:
                continue
            ur = x[j] * V[i, 0] + y[j] * V[i, 1] - p[j]
            ut = x[i] * V[i, 0] + y[i] * V[i, 1] - p[i]
            if ur - ut > 1e-7:
                ic_ok = False
                break
        if not ic_ok:
            break

    if not ic_ok:
        # Try to fix small violations by reducing payments slightly
        # Reduce all payments by a small epsilon to ensure feasibility
        eps = 1e-6
        p_adjusted = np.maximum(p - eps, 0.0)
        # Re-check
        ic_ok2 = True
        for i in range(N):
            if total_val[i] - p_adjusted[i] < -1e-7:
                ic_ok2 = False
                break
            for j in range(N):
                if i == j:
                    continue
                ur = x[j] * V[i, 0] + y[j] * V[i, 1] - p_adjusted[j]
                ut = x[i] * V[i, 0] + y[i] * V[i, 1] - p_adjusted[i]
                if ur - ut > 1e-7:
                    ic_ok2 = False
                    break
            if not ic_ok2:
                break
        
        if ic_ok2:
            p = p_adjusted
        else:
            return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    return {'allocations': np.column_stack((x, y)), 'payments': p}
