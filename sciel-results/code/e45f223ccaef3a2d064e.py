import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Parse problem structure
    values = [0, 1, 2, 3]
    
    # Get distribution
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
        # Ensure no negative probs
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    # Determine types
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n_types = len(types)
    if n_types != 16:
        # Fallback if support parsing is weird, use standard grid
        types = [(i, j) for i in values for j in values]
        n_types = 16
    
    type_to_vals = {idx: (v1, v2) for idx, (v1, v2) in enumerate(types)}
    type_vals = np.array([type_to_vals[i] for i in range(16)], dtype=float)
    v1_all = type_vals[:, 0]
    v2_all = type_vals[:, 1]
    
    # LP Formulation
    # Variables: x1[0..15], x2[16..31], p[32..47]
    n_vars = 48
    
    A_rows = []
    b_rows = []
    
    # IC: p(t) - p(t') <= v1(t)*(x1(t) - x1(t')) + v2(t)*(x2(t) - x2(t'))
    # => -v1(t)*x1(t) + v1(t)*x1(t') - v2(t)*x2(t) + v2(t)*x2(t') + p(t) - p(t') <= 0
    for t in range(16):
        for t2 in range(16):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            row[t] = -v1_all[t]
            row[t2] = v1_all[t]
            row[16 + t] = -v2_all[t]
            row[16 + t2] = v2_all[t]
            row[32 + t] = 1.0
            row[32 + t2] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR: p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    # => -v1(t)*x1(t) - v2(t)*x2(t) + p(t) <= 0
    for t in range(16):
        row = np.zeros(n_vars)
        row[t] = -v1_all[t]
        row[16 + t] = -v2_all[t]
        row[32 + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Objective: maximize sum p(t) * probs[t]
    c = np.zeros(n_vars)
    c[32:48] = -probs
    
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16
    
    # Solve
    result = None
    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-14,
            'primal_feasibility_tolerance': 1e-14,
            'time_limit': 70
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        pass

    if result is not None and result.success:
        x = result.x
        x1 = np.clip(x[0:16], 0, 1)
        x2 = np.clip(x[16:32], 0, 1)
        p = np.maximum(x[32:48], 0)
        allocations = np.column_stack((x1, x2))
        payments = p
        
        # Strict Verification
        # Calculate max regret
        max_regret = 0.0
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            u_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            for t2 in range(16):
                if t == t2:
                    continue
                u_t2_as_t = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
                if u_t < u_t2_as_t:
                    viol = u_t2_as_t - u_t
                    if viol > max_regret:
                        max_regret = viol
            # IR check
            if u_t < 0:
                viol = -u_t
                if viol > max_regret:
                    max_regret = viol
        
        # If regret is within tolerance, accept
        if max_regret <= 1e-8:
            return {'allocations': allocations, 'payments': payments}
        else:
            # If LP solution is not feasible enough, fall back to zero
            # This is safer than trying to fix it, as fixing might break other constraints
            # and the zero mechanism is always feasible.
            pass
    else:
        pass

    # Fallback: Zero mechanism
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    return {'allocations': allocations, 'payments': payments}
