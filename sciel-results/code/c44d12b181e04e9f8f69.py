"""Optimal finite-support mechanism via reduced LP using payment determination by allocation."""
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Parse distribution
    values = [0, 1, 2, 3]
    
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
    
    # Parse support
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
    
    n = 16
    assert len(types) == n, f"Expected 16 types, got {len(types)}"
    
    # Value matrix: val[t] = (v1, v2)
    val_matrix = np.array(types, dtype=float)  # shape (16, 2)
    
    # For a single additive bidder, IC implies p[t] = v[t]·x[t] - v[0]·x[0]
    # (fixing p[0] = 0 WLOG, which is valid since type 0 has value (0,0))
    # Objective: maximize sum_t probs[t] * (v[t]·x[t] - v[0]·x[0])
    # Constraints: x[t,g] in [0,1], and p[t] >= 0 i.e. v[t]·x[t] >= v[0]·x[0]
    
    n_vars = n * 2  # only allocation variables
    
    # Objective: maximize sum_t probs[t] * (v[t]·x[t] - v[0]·x[0])
    # = sum_t probs[t] * v[t,0]*x[t,0] + probs[t]*v[t,1]*x[t,1] - probs[t]*v[0,0]*x[0,0] - probs[t]*v[0,1]*x[0,1]
    # Since v[0] = (0,0) for type 0 (first type in sorted order), the last terms vanish
    # But let's be general in case type 0 is not (0,0)
    
    c = np.zeros(n_vars)
    v0_dot = val_matrix[0, 0] * val_matrix[0, 0] + val_matrix[0, 1] * val_matrix[0, 1]  # wait, this is wrong
    # v[0]·x[0] = val_matrix[0,0]*x[0,0] + val_matrix[0,1]*x[0,1]
    # So the coefficient for x[t,g] from the -sum_t probs[t]*v[0]·x[0] term is:
    # -probs[t] * val_matrix[0,g] for the x[0,g] variables
    
    # Let me recompute:
    # Objective = sum_t probs[t] * (sum_g val_matrix[t,g]*x[t,g] - sum_g val_matrix[0,g]*x[0,g])
    # Coefficient of x[t,g] = probs[t]*val_matrix[t,g] - (sum over all t' of probs[t']*val_matrix[0,g] if t==0 else 0)
    # Wait no: the -sum_g val_matrix[0,g]*x[0,g] term is multiplied by probs[t] for each t
    # So total coefficient of x[0,g] = sum_t probs[t]*val_matrix[0,g] ... no wait
    
    # Let me be more careful:
    # Rev = sum_t probs[t] * (v[t]·x[t] - v[0]·x[0])
    # = sum_t probs[t] * sum_g v[t,g]*x[t,g] - sum_t probs[t] * sum_g v[0,g]*x[0,g]
    # = sum_t sum_g probs[t]*v[t,g]*x[t,g] - sum_g (sum_t probs[t]) * v[0,g]*x[0,g]
    # = sum_t sum_g probs[t]*v[t,g]*x[t,g] - sum_g v[0,g]*x[0,g]  (since sum_t probs[t] = 1)
    
    # So coefficient of x[t,g] = probs[t]*val_matrix[t,g] - (val_matrix[0,g] if t==0 else 0)
    
    for t in range(n):
        for g in range(2):
            c[t*2 + g] = -probs[t] * val_matrix[t, g]  # negating for minimization
            if t == 0:
                c[t*2 + g] -= -val_matrix[0, g]  # subtract the -val_matrix[0,g] term, i.e. add val_matrix[0,g]
                # Wait: c should be such that we minimize c·x = -Rev
                # Rev coeff of x[t,g] = probs[t]*val_matrix[t,g] - (val_matrix[0,g] if t==0 else 0)
                # So c[t*2+g] = -(probs[t]*val_matrix[t,g] - (val_matrix[0,g] if t==0 else 0))
                # = -probs[t]*val_matrix[t,g] + (val_matrix[0,g] if t==0 else 0)
                c[t*2 + g] = -probs[t] * val_matrix[t, g] + val_matrix[0, g]
    
    # IC constraints for the payment rule p[t] = v[t]·x[t] - v[0]·x[0]:
    # For all t, t': v[t']·x[t'] >= v[t]·x[t']  =>  (v[t'] - v[t])·x[t'] >= 0
    # We need: v[t']·x[t'] - v[t]·x[t'] >= 0 for all t != t'
    A_rows = []
    b_rows = []
    
    for t_prime in range(n):
        for t in range(n):
            if t == t_prime:
                continue
            row = np.zeros(n_vars)
            # (v[t'] - v[t])·x[t'] >= 0  =>  -(v[t'] - v[t])·x[t'] <= 0
            for g in range(2):
                row[t_prime*2 + g] = -(val_matrix[t_prime, g] - val_matrix[t, g])
            A_rows.append(row)
            b_rows.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    # Bounds: x[t,g] in [0,1]
    bounds = [(0, 1)] * n_vars
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12,
                                  'dual_stall_tolerance': 1e-12,
                                  'primal_stall_tolerance': 1e-12})
        if result.success:
            alloc = result.x[:n*2].reshape(n, 2)
            # Compute payments: p[t] = v[t]·x[t] - v[0]·x[0]
            pay = np.array([np.dot(val_matrix[t], alloc[t]) - np.dot(val_matrix[0], alloc[0]) for t in range(n)])
            
            # Clean up numerical noise
            alloc = np.clip(alloc, 0, 1)
            pay = np.maximum(pay, 0)
            
            # Verify IC and IR constraints
            # IC: for all t, t2: v[t]·x[t] - p[t] >= v[t]·x[t2] - p[t2]
            ic_ok = True
            for t in range(n):
                for t2 in range(n):
                    util_t = np.dot(val_matrix[t], alloc[t]) - pay[t]
                    util_t2 = np.dot(val_matrix[t], alloc[t2]) - pay[t2]
                    if util_t < util_t2 - 1e-8:
                        ic_ok = False
                        break
                if not ic_ok:
                    break
            
            # IR: p[t] >= 0 and p[t] <= v[t]·x[t]
            ir_ok = all(pay[t] <= np.dot(val_matrix[t], alloc[t]) + 1e-8 for t in range(n))
            
            if ic_ok and ir_ok:
                return {'allocations': alloc, 'payments': pay}
            else:
                # Fallback: try the original full LP
                return _full_lp_fallback(probs, val_matrix, n)
        else:
            return _full_lp_fallback(probs, val_matrix, n)
    except Exception:
        return _full_lp_fallback(probs, val_matrix, n)

def _full_lp_fallback(probs, val_matrix, n):
    """Fallback to the full LP formulation with explicit payment variables."""
    n_vars = n * 2 + n
    
    # Maximize sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n):
        c[n*2 + t] = -probs[t]
    
    A_rows = []
    b_rows = []
    
    # IC: v[t]·x[t] - p[t] >= v[t]·x[t2] - p[t2] for all t, t2
    # => v[t]·x[t] - v[t]·x[t2] - p[t] + p[t2] >= 0
    # => -(v[t]·x[t] - v[t]·x[t2] - p[t] + p[t2]) <= 0
    for t in range(n):
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            row[n*2 + t] = -1.0
            row[n*2 + t2] = 1.0
            for g in range(2):
                row[t*2 + g] = -val_matrix[t, g]
                row[t2*2 + g] = val_matrix[t, g]
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR: p[t] <= v[t]·x[t]  =>  p[t] - v[t]·x[t] <= 0
    for t in range(n):
        row = np.zeros(n_vars)
        row[n*2 + t] = 1.0
        for g in range(2):
            row[t*2 + g] = -val_matrix[t, g]
        A_rows.append(row)
        b_rows.append(0.0)
        
    # Non-negative payments: p[t] >= 0  =>  -p[t] <= 0
    for t in range(n):
        row = np.zeros(n_vars)
        row[n*2 + t] = -1.0
        A_rows.append(row)
        b_rows.append(0.0)
        
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    bounds = []
    for t in range(n):
        for g in range(2):
            bounds.append((0, 1))
    for t in range(n):
        bounds.append((0, None))
        
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12})
        if result.success:
            alloc = result.x[:n*2].reshape(n, 2)
            pay = result.x[n*2:n*2+n]
            alloc = np.clip(alloc, 0, 1)
            pay = np.maximum(pay, 0)
            
            # Verify
            ic_ok = True
            for t in range(n):
                for t2 in range(n):
                    util_t = np.dot(val_matrix[t], alloc[t]) - pay[t]
                    util_t2 = np.dot(val_matrix[t], alloc[t2]) - pay[t2]
                    if util_t < util_t2 - 1e-8:
                        ic_ok = False
                        break
                if not ic_ok:
                    break
            ir_ok = all(pay[t] <= np.dot(val_matrix[t], alloc[t]) + 1e-8 and pay[t] >= -1e-8 for t in range(n))
            
            if ic_ok and ir_ok:
                return {'allocations': alloc, 'payments': pay}
            else:
                return {'allocations': np.zeros((n, 2)), 'payments': np.zeros(n)}
        else:
            return {'allocations': np.zeros((n, 2)), 'payments': np.zeros(n)}
    except Exception:
        return {'allocations': np.zeros((n, 2)), 'payments': np.zeros(n)}
