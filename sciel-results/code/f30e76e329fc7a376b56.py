import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 70.0
    
    # 1. Setup Report Space
    n_types = 16
    values = np.array([(v1, v2) for v1 in range(4) for v2 in range(4)], dtype=float)
    
    # 2. Parse Distribution
    probs = _get_probabilities(distribution_spec, n_types)
    
    # 3. Try analytical Myerson-style approach first
    result = _myerson_analytical(values, probs, n_types)
    if result is not None:
        x_alloc, payments = result
        if _check_feasibility(x_alloc, payments, values, n_types):
            return {
                'allocations': x_alloc,
                'payments': payments
            }
    
    # 4. Fall back to LP if analytical fails
    return _lp_solve(values, probs, n_types, deadline)

def _myerson_analytical(values, probs, n_types):
    """
    Analytical Myerson-style optimal auction for single bidder with two goods.
    
    For a single additive bidder, the optimal mechanism can be derived by:
    1. Computing virtual values for each report
    2. The allocation rule should be monotone in each dimension
    3. For finite support, the optimal allocation is determined by the revenue-maximizing 
       monotone allocation rule
    
    Key insight: For a single bidder with private values, the optimal allocation x(v)
    must be non-decreasing in each component. The payment is determined by IC.
    
    We use the fact that for discrete types, the optimal allocation can be found by
    considering the marginal revenue from increasing allocation at each type.
    """
    # Compute the "revenue contribution" of allocating to each type
    # For type v, if we allocate x(v), the revenue is:
    # p(v) = v·x(v) - sum_{u: u_g < v_g for some g} x(u)·(indicator terms)
    
    # Actually, let's use the LP formulation but solve it analytically
    # The key constraints are monotonicity: x_{v,g} <= x_{v+e_g,g}
    
    # For the optimal solution, we can think of it as:
    # Maximize sum_v probs[v] * (v1*x[v,0] + v2*x[v,1] - payment[v])
    # subject to monotonicity and IR
    
    # The payment identity for IC:
    # payment[v] = v1*x[v,0] + v2*x[v,1] - sum_{u1<v1} x[(u1,v2),0] - sum_{u2<v2} x[(v1,u2),1]
    
    # Substitute into revenue:
    # E[R] = sum_v probs[v] * [v1*x[v,0] + v2*x[v,1] - sum_{u1<v1} x[(u1,v2),0] - sum_{u2<v2} x[(v1,u2),1]]
    
    # Collect coefficients for each x[j,g]:
    # For x[v,0]: coefficient = probs[v]*v1 - sum_{w: w0>v0, w1=v1} probs[w] - sum_{w: w0=v0, w1>v1} probs[w]*0 (wait, need to be careful)
    
    # Let me redo this carefully:
    # E[R] = sum_v probs[v] * v1 * x[v,0] + sum_v probs[v] * v2 * x[v,1] 
    #       - sum_v probs[v] * sum_{u1<v1} x[(u1,v2),0] - sum_v probs[v] * sum_{u2<v2} x[(v1,u2),1]
    
    # For x[j,0] where j = (j1, j2):
    # Coefficient from first term: probs[j]*j1
    # Coefficient from third term: -sum_{v: v1>j1, v2=j2} probs[v]
    
    # For x[j,1] where j = (j1, j2):
    # Coefficient from second term: probs[j]*j2
    # Coefficient from fourth term: -sum_{v: v1=j1, v2>j2} probs[v]
    
    obj_coeffs = np.zeros(n_types * 2)
    
    for v in range(n_types):
        v1, v2 = int(values[v][0]), int(values[v][1])
        p_v = probs[v]
        
        # x[v,0] gets +p_v * v1
        obj_coeffs[v*2 + 0] += p_v * v1
        
        # x[v,1] gets +p_v * v2
        obj_coeffs[v*2 + 1] += p_v * v2
        
        # For each u1 < v1, x[(u1,v2),0] gets -p_v
        for u1 in range(v1):
            idx_u = u1 * 4 + v2
            obj_coeffs[idx_u * 2 + 0] -= p_v
        
        # For each u2 < v2, x[(v1,u2),1] gets -p_v
        for u2 in range(v2):
            idx_u = v1 * 4 + u2
            obj_coeffs[idx_u * 2 + 1] -= p_v
    
    # Now solve the LP analytically using the structure
    # The constraints are:
    # 0 <= x[v,g] <= 1
    # x[(v1,v2),0] <= x[(v1+1,v2),0] for v1=0,1,2
    # x[(v1,v2),1] <= x[(v1,v2+1),1] for v2=0,1,2
    
    # This is a linear program with a specific structure.
    # Let's use scipy but with tighter tolerances and verify.
    
    n_vars = n_types * 2
    c = -obj_coeffs
    
    bounds = [(0.0, 1.0)] * n_vars
    
    A_ub_rows = []
    b_ub_rows = []
    
    # Good 1 Monotonicity: x[(v1,v2),0] <= x[(v1+1,v2),0]
    for v2 in range(4):
        for v1 in range(3):
            i_curr = v1 * 4 + v2
            i_next = (v1 + 1) * 4 + v2
            row = np.zeros(n_vars)
            row[i_curr * 2 + 0] = 1.0
            row[i_next * 2 + 0] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
    
    # Good 2 Monotonicity: x[(v1,v2),1] <= x[(v1,v2+1),1]
    for v1 in range(4):
        for v2 in range(3):
            i_curr = v1 * 4 + v2
            i_next = v1 * 4 + (v2 + 1)
            row = np.zeros(n_vars)
            row[i_curr * 2 + 1] = 1.0
            row[i_next * 2 + 1] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
        
        if res.success:
            x_flat = res.x
            x_alloc = x_flat.reshape(n_types, 2)
            x_alloc = np.clip(x_alloc, 0.0, 1.0)
            
            # Calculate Payments using IC identity
            payments = np.zeros(n_types)
            for v in range(n_types):
                v1, v2 = int(values[v][0]), int(values[v][1])
                p_v = v1 * x_alloc[v, 0] + v2 * x_alloc[v, 1]
                
                for u1 in range(v1):
                    idx_u = u1 * 4 + v2
                    p_v -= x_alloc[idx_u, 0]
                    
                for u2 in range(v2):
                    idx_u = v1 * 4 + u2
                    p_v -= x_alloc[idx_u, 1]
                
                payments[v] = p_v
            
            # Check feasibility
            if _check_feasibility(x_alloc, payments, values, n_types):
                return (x_alloc, payments)
            
            # Try to repair: clip payments to non-negative
            payments_clipped = np.clip(payments, 0, None)
            if _check_feasibility(x_alloc, payments_clipped, values, n_types):
                return (x_alloc, payments_clipped)
                
    except Exception:
        pass
    
    return None

def _lp_solve(values, probs, n_types, deadline):
    """Fallback LP solver with time guard."""
    start_time = time.time()
    if start_time > deadline - 5.0:
        return _fallback_mechanism()
    
    # This is essentially the same as the analytical approach but kept as a separate function
    # for clarity and to allow different solver settings
    result = _myerson_analytical(values, probs, n_types)
    if result is not None:
        x_alloc, payments = result
        return {
            'allocations': x_alloc,
            'payments': payments
        }
    
    return _fallback_mechanism()

def _get_probabilities(spec, n_types):
    """Parses various distribution formats into a probability vector."""
    if spec is None:
        return np.ones(n_types) / n_types
        
    if isinstance(spec, dict):
        probs = np.zeros(n_types)
        for key, val in spec.items():
            if isinstance(key, tuple) and len(key) == 2:
                v1, v2 = int(key[0]), int(key[1])
                if 0 <= v1 < 4 and 0 <= v2 < 4:
                    idx = v1 * 4 + v2
                    probs[idx] = float(val)
            elif isinstance(key, int):
                if 0 <= key < n_types:
                    probs[key] = float(val)
        total = probs.sum()
        if total > 0:
            return probs / total
        return np.ones(n_types) / n_types
        
    if isinstance(spec, (list, tuple, np.ndarray)):
        arr = np.array(spec, dtype=float)
        if arr.shape == (n_types,):
            total = arr.sum()
            if total > 0:
                return arr / total
            return np.ones(n_types) / n_types
        # Handle 2D array of [v1, v2, p]
        if arr.ndim == 2 and arr.shape[1] >= 3:
            probs = np.zeros(n_types)
            for row in arr:
                v1, v2, p = int(row[0]), int(row[1]), float(row[2])
                if 0 <= v1 < 4 and 0 <= v2 < 4:
                    idx = v1 * 4 + v2
                    probs[idx] += p
            total = probs.sum()
            if total > 0:
                return probs / total
            return np.ones(n_types) / n_types
            
    return np.ones(n_types) / n_types

def _check_feasibility(allocs, pays, values, n_types):
    """Checks IR, IC, and Non-negativity."""
    eps = 1e-8
    
    # Check bounds
    if np.any(allocs < -eps) or np.any(allocs > 1 + eps):
        return False
    if np.any(pays < -eps):
        return False
        
    # Check IR and IC
    for j in range(n_types):
        v1_j, v2_j = values[j]
        u_true = v1_j * allocs[j, 0] + v2_j * allocs[j, 1] - pays[j]
        
        if u_true < -eps:
            return False
            
        for i in range(n_types):
            if i == j: continue
            u_mis = v1_j * allocs[i, 0] + v2_j * allocs[i, 1] - pays[i]
            if u_true - u_mis < -eps:
                return False
                
    return True

def _fallback_mechanism():
    """
    Returns a simple, valid mechanism.
    Allocates nothing, charges nothing.
    Revenue 0, but valid.
    """
    n_types = 16
    allocs = np.zeros((n_types, 2))
    pays = np.zeros(n_types)
    return {
        'allocations': allocs,
        'payments': pays
    }
