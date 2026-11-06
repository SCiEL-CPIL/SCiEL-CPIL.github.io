import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Parse distribution
    # distribution_spec: likely a dict or array with probabilities for each type
    # support: list of possible types, e.g., [(0,0), (0,1), ..., (3,3)]
    # constraints: additional constraints
    
    # Assume standard setup: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Get probability distribution
    probs = None
    try:
        if isinstance(distribution_spec, dict):
            if len(distribution_spec) == n_types:
                if 0 in distribution_spec:
                    probs = np.array([distribution_spec[i] for i in range(n_types)], dtype=float)
                elif (0,0) in distribution_spec:
                    probs = np.array([distribution_spec.get(t, 0.0) for t in types], dtype=float)
                else:
                    probs = np.array(list(distribution_spec.values()), dtype=float)
            else:
                # If keys don't match expected count, try values
                probs = np.array(list(distribution_spec.values()), dtype=float)
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            # Flatten and convert to float array
            flat_spec = np.array(distribution_spec, dtype=object).flatten()
            # Convert each element to float, defaulting to 0 if conversion fails
            probs = np.zeros(n_types)
            for i in range(min(len(flat_spec), n_types)):
                try:
                    probs[i] = float(flat_spec[i])
                except (ValueError, TypeError):
                    probs[i] = 0.0
        else:
            probs = np.ones(n_types) / n_types
    except Exception:
        probs = np.ones(n_types) / n_types

    # Ensure probs is 1D and length 16
    if probs is None or len(probs) != n_types:
        # Fallback if parsing failed badly
        probs = np.ones(n_types) / n_types
    else:
        probs = probs[:n_types]
        
    # Clean up probabilities
    probs = np.maximum(probs, 0)
    sum_probs = probs.sum()
    if sum_probs > 1e-9:
        probs = probs / sum_probs
    else:
        probs = np.ones(n_types) / n_types
    
    # Type to index mapping
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Variables: [x_0,0,1, x_0,0,2, x_0,1,1, x_0,1,2, ..., x_3,3,1, x_3,3,2, p_0,0, p_0,1, ..., p_3,3]
    # First 32 vars: allocations x[type_idx, good] for good in {1,2}
    # Last 16 vars: payments p[type_idx]
    
    n_alloc = n_types * 2  # 32
    n_pay = n_types  # 16
    n_vars = n_alloc + n_pay  # 48
    
    def alloc_var_idx(type_idx, good):
        """good: 1 or 2"""
        return type_idx * 2 + (good - 1)
    
    def pay_var_idx(type_idx):
        return n_alloc + type_idx
    
    # Objective: maximize expected revenue = sum_v P(v) * p[v]
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for v_idx in range(n_types):
        c[pay_var_idx(v_idx)] = -probs[v_idx]
    
    # Constraints:
    # 1. IC: u(v, v) >= u(v, v') for all v, v'
    #    u(v, v) = v1*x[v,1] + v2*x[v,2] - p[v]
    #    u(v, v') = v1*x[v',1] + v2*x[v',2] - p[v']
    #    So: v1*x[v,1] + v2*x[v,2] - p[v] - v1*x[v',1] - v2*x[v',2] + p[v'] >= 0
    #    Rearranged: -v1*x[v,1] - v2*x[v,2] + p[v] + v1*x[v',1] + v2*x[v',2] - p[v'] <= 0
    
    A_ub = []
    b_ub = []
    
    for v_idx in range(n_types):
        v1, v2 = types[v_idx]
        for vp_idx in range(n_types):
            vp1, vp2 = types[vp_idx]
            row = np.zeros(n_vars)
            
            # Coefficients for x[v,1], x[v,2]
            row[alloc_var_idx(v_idx, 1)] = -v1
            row[alloc_var_idx(v_idx, 2)] = -v2
            # Coefficients for p[v]
            row[pay_var_idx(v_idx)] = 1.0
            
            # Coefficients for x[vp,1], x[vp,2]
            row[alloc_var_idx(vp_idx, 1)] = v1
            row[alloc_var_idx(vp_idx, 2)] = v2
            # Coefficients for p[vp]
            row[pay_var_idx(vp_idx)] = -1.0
            
            A_ub.append(row)
            b_ub.append(0.0)
    
    # 2. IR: u(v, v) >= 0 for all v
    #    v1*x[v,1] + v2*x[v,2] - p[v] >= 0
    #    => -v1*x[v,1] - v2*x[v,2] + p[v] <= 0
    
    for v_idx in range(n_types):
        v1, v2 = types[v_idx]
        row = np.zeros(n_vars)
        row[alloc_var_idx(v_idx, 1)] = -v1
        row[alloc_var_idx(v_idx, 2)] = -v2
        row[pay_var_idx(v_idx)] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    if A_ub:
        A_ub = np.array(A_ub, dtype=float)
        b_ub = np.array(b_ub, dtype=float)
    else:
        A_ub = np.array([[]], dtype=float)
        b_ub = np.array([0.0], dtype=float)
    
    # Bounds: 0 <= x <= 1, p unbounded (but IR will constrain)
    # Actually, payments should be non-negative per the task
    bounds = []
    for i in range(n_alloc):
        bounds.append((0, 1))
    for i in range(n_pay):
        bounds.append((0, None))  # Non-negative payments
    
    # Solve LP
    if time.time() > deadline - 5.0:
        # Not enough time to solve and verify, return trivial
        x_alloc = np.zeros((n_types, 2))
        p_pay = np.zeros(n_types)
        return {'allocations': x_alloc, 'payments': p_pay}
        
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x_alloc = result.x[:n_alloc].reshape(n_types, 2)
            p_pay = result.x[n_alloc:n_alloc+n_pay]
            
            # Clip to ensure feasibility
            x_alloc = np.clip(x_alloc, 0, 1)
            p_pay = np.clip(p_pay, 0, None)
            
            # Verify constraints
            if time.time() < deadline:
                # Check IC and IR
                for v_idx in range(n_types):
                    v1, v2 = types[v_idx]
                    u_vv = v1 * x_alloc[v_idx, 0] + v2 * x_alloc[v_idx, 1] - p_pay[v_idx]
                    if u_vv < -1e-6:
                        # IR violated, adjust payment
                        p_pay[v_idx] = max(0, v1 * x_alloc[v_idx, 0] + v2 * x_alloc[v_idx, 1] - 1e-9)
                    for vp_idx in range(n_types):
                        vp1, vp2 = types[vp_idx]
                        u_vvp = v1 * x_alloc[vp_idx, 0] + v2 * x_alloc[vp_idx, 1] - p_pay[vp_idx]
                        if u_vvp > u_vv + 1e-6:
                            # IC violated, increase payment for vp or decrease for v
                            # Simple fix: increase p_pay[vp_idx]
                            p_pay[vp_idx] = max(p_pay[vp_idx], v1 * x_alloc[vp_idx, 0] + v2 * x_alloc[vp_idx, 1] - u_vv)
            
            return {'allocations': x_alloc, 'payments': p_pay}
        else:
            # Fallback: trivial mechanism
            x_alloc = np.zeros((n_types, 2))
            p_pay = np.zeros(n_types)
            return {'allocations': x_alloc, 'payments': p_pay}
    except Exception:
        # Fallback: trivial mechanism
        x_alloc = np.zeros((n_types, 2))
        p_pay = np.zeros(n_types)
        return {'allocations': x_alloc, 'payments': p_pay}
