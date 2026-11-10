import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # 16 types: (v1, v2) with v1, v2 in {0,1,2,3}
    # Index: v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n = 16

    # Parse distribution
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n:
            probs = np.ones(n) / n
    else:
        probs = np.ones(n) / n

    total = probs.sum()
    if total <= 0 or not np.isfinite(total):
        probs = np.ones(n) / n
    else:
        probs = probs / total

    # LP variables:
    # x[0:16]   = q1[t]  (allocation prob for good 1 to type t)
    # x[16:32]  = q2[t]  (allocation prob for good 2 to type t)
    # x[32:48]  = p[t]   (payment for type t)

    # Objective: maximize sum_t probs[t] * p[t]
    # => minimize -sum_t probs[t] * p[t]
    c = np.zeros(48)
    for t in range(n):
        c[32 + t] = -probs[t]

    A_ub = []
    b_ub = []

    # IC constraints: for each ordered pair (t, t'), t != t'
    # p[t] - p[t'] - v1[t]*q1[t] + v1[t]*q1[t'] - v2[t]*q2[t] + v2[t]*q2[t'] <= 0
    for t in range(n):
        v1t, v2t = types[t]
        for tp in range(n):
            if tp == t:
                continue
            row = np.zeros(48)
            row[t] = -v1t
            row[tp] = v1t
            row[16 + t] = -v2t
            row[16 + tp] = v2t
            row[32 + t] = 1.0
            row[32 + tp] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # IR constraints: p[t] - v1[t]*q1[t] - v2[t]*q2[t] <= 0
    for t in range(n):
        v1t, v2t = types[t]
        row = np.zeros(48)
        row[t] = -v1t
        row[16 + t] = -v2t
        row[32 + t] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Nonnegative payments: -p[t] <= 0
    for t in range(n):
        row = np.zeros(48)
        row[32 + t] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Bounds
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Check time before solving
    if time.time() > deadline - 5:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    # Solve with tight tolerance
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'time_limit': 30, 'mip_rel_gap': 0,
                                  'presolve_mip_gap': 0, 'primal_feasibility_tolerance': 1e-12,
                                  'dual_feasibility_tolerance': 1e-12})
    except Exception:
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception:
            return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if time.time() > deadline - 2:
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    if not (result.success and result.x is not None):
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

    q1 = np.clip(result.x[0:16], 0, 1)
    q2 = np.clip(result.x[16:32], 0, 1)
    p = np.clip(result.x[32:48], 0, None)
    
    x_init = np.concatenate([q1, q2, p])

    # Polish: Solve the system of active constraints to refine the vertex
    def polish(x):
        # Identify active constraints
        active_rows = []
        active_rhs = []
        
        # Inequality constraints A_ub x <= b_ub
        res = A_ub @ x - b_ub
        for i in range(len(res)):
            if res[i] > -1e-9:
                active_rows.append(A_ub[i])
                active_rhs.append(b_ub[i])
                
        # Bounds
        lower_bounds = [0.0] * 48
        upper_bounds = [1.0] * 16 + [1.0] * 16 + [np.inf] * 16
        
        for i in range(48):
            if x[i] < lower_bounds[i] + 1e-9:
                row = np.zeros(48)
                row[i] = -1.0
                active_rows.append(row)
                active_rhs.append(-lower_bounds[i])
            if np.isfinite(upper_bounds[i]) and x[i] > upper_bounds[i] - 1e-9:
                row = np.zeros(48)
                row[i] = 1.0
                active_rows.append(row)
                active_rhs.append(upper_bounds[i])
                
        if len(active_rows) < 48:
            return x
            
        M = np.array(active_rows)
        v = np.array(active_rhs)
        
        try:
            sol, residuals, rank, s = np.linalg.lstsq(M, v, rcond=None)
            # Check feasibility of sol
            if np.any(sol < -1e-8):
                return x
            if np.any(sol[0:32] > 1 + 1e-8):
                return x
            res_new = A_ub @ sol - b_ub
            if np.any(res_new > 1e-8):
                return x
                
            obj_old = c @ x
            obj_new = c @ sol
            # Minimize c @ x
            if obj_new <= obj_old + 1e-10:
                return sol
        except Exception:
            pass
            
        return x

    x_polished = polish(x_init)
    
    q1 = np.clip(x_polished[0:16], 0, 1)
    q2 = np.clip(x_polished[16:32], 0, 1)
    p = np.clip(x_polished[32:48], 0, None)

    # Final feasibility check and minimal fix if needed
    def check_feasibility(q1, q2, p):
        max_viol = 0.0
        for t in range(n):
            v1t, v2t = types[t]
            if p[t] - v1t * q1[t] - v2t * q2[t] > max_viol:
                max_viol = p[t] - v1t * q1[t] - v2t * q2[t]
            if -p[t] > max_viol:
                max_viol = -p[t]
            if q1[t] < -max_viol:
                max_viol = -q1[t]
            if q1[t] > 1 + max_viol:
                max_viol = q1[t] - 1
            if q2[t] < -max_viol:
                max_viol = -q2[t]
            if q2[t] > 1 + max_viol:
                max_viol = q2[t] - 1
            for tp in range(n):
                if tp == t:
                    continue
                util_t = v1t * q1[t] + v2t * q2[t] - p[t]
                util_tp = v1t * q1[tp] + v2t * q2[tp] - p[tp]
                if util_tp - util_t > max_viol:
                    max_viol = util_tp - util_t
        return max_viol

    viol = check_feasibility(q1, q2, p)
    if viol > 1e-8:
        for iteration in range(10):
            changed = False
            for t in range(n):
                v1t, v2t = types[t]
                util_t = v1t * q1[t] + v2t * q2[t] - p[t]
                max_util_other = -np.inf
                for tp in range(n):
                    if tp == t:
                        continue
                    util_tp = v1t * q1[tp] + v2t * q2[tp] - p[tp]
                    if util_tp > max_util_other:
                        max_util_other = util_tp
                if max_util_other - util_t > 1e-8:
                    new_p_t = v1t * q1[t] + v2t * q2[t] - max_util_other
                    if new_p_t < p[t]:
                        p[t] = max(0.0, new_p_t)
                        changed = True
            if not changed:
                break
        p = np.clip(p, 0, None)
        q1 = np.clip(q1, 0, 1)
        q2 = np.clip(q2, 0, 1)

    allocations = np.column_stack([q1, q2])
    payments = p

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
