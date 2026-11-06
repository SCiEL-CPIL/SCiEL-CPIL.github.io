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
    deadline = start_time + 85.0  # 85s safety margin for 90s limit

    # 1. Setup Report Space
    n_types = 16
    values = np.array([(v1, v2) for v1 in range(4) for v2 in range(4)], dtype=float)
    
    # 2. Parse Distribution
    probs = _get_probabilities(distribution_spec, n_types)
    
    # 3. Solve the Reduced LP with Non-Negative Payment Constraints
    result = _solve_reduced_lp(values, probs, n_types, deadline)
    
    if result is not None:
        x_alloc, payments = result
        # Final verification
        if _check_feasibility(x_alloc, payments, values, n_types):
            return {
                'allocations': x_alloc,
                'payments': payments
            }
        
        # If verification fails, try to repair or fallback
        return _fallback_mechanism()
    
    return _fallback_mechanism()

def _solve_reduced_lp(values, probs, n_types, deadline):
    """
    Solves the optimal mechanism problem using the reduced LP formulation.
    """
    start_time = time.time()
    if start_time > deadline - 5.0:
        return None

    n_vars = n_types * 2  # 32 variables
    
    # 1. Objective Coefficients
    obj_coeffs = np.zeros(n_vars)
    
    for v in range(n_types):
        v1, v2 = int(values[v][0]), int(values[v][1])
        p_v = probs[v]
        
        obj_coeffs[v*2 + 0] += p_v * v1
        obj_coeffs[v*2 + 1] += p_v * v2
        
        for u1 in range(v1):
            idx_u = u1 * 4 + v2
            obj_coeffs[idx_u * 2 + 0] -= p_v
            
        for u2 in range(v2):
            idx_u = v1 * 4 + u2
            obj_coeffs[idx_u * 2 + 1] -= p_v

    c = -obj_coeffs
    bounds = [(0.0, 1.0)] * n_vars

    A_ub_rows = []
    b_ub_rows = []
    
    # Monotonicity
    for v2 in range(4):
        for v1 in range(3):
            i_curr = v1 * 4 + v2
            i_next = (v1 + 1) * 4 + v2
            row = np.zeros(n_vars)
            row[i_curr * 2 + 0] = 1.0
            row[i_next * 2 + 0] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
    for v1 in range(4):
        for v2 in range(3):
            i_curr = v1 * 4 + v2
            i_next = v1 * 4 + (v2 + 1)
            row = np.zeros(n_vars)
            row[i_curr * 2 + 1] = 1.0
            row[i_next * 2 + 1] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
    # Non-Negative Payments
    for v in range(n_types):
        v1, v2 = int(values[v][0]), int(values[v][1])
        row = np.zeros(n_vars)
        
        if v1 > 0:
            row[v*2 + 0] = -v1
        if v2 > 0:
            row[v*2 + 1] = -v2
            
        for u1 in range(v1):
            idx_u = u1 * 4 + v2
            row[idx_u * 2 + 0] += 1.0
            
        for u2 in range(v2):
            idx_u = v1 * 4 + u2
            row[idx_u * 2 + 1] += 1.0
            
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
        
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12, 'time_limit': 70})
        
        if not res.success:
            return None
            
        x_flat = res.x
        x_alloc = x_flat.reshape(n_types, 2)
        
        # Polish: Snap values close to 0 or 1
        x_alloc = _polish_solution(x_alloc, values, probs, n_types)
        
        # Calculate Payments
        payments = _calculate_payments(x_alloc, values, n_types)
        payments = np.clip(payments, 0.0, None)
        
        return (x_alloc, payments)
        
    except Exception:
        return None

def _polish_solution(x_alloc, values, probs, n_types):
    """
    Snaps allocation probabilities close to 0 or 1 to exact values to improve numerical stability.
    """
    x_polished = x_alloc.copy()
    tol = 1e-9
    
    for i in range(n_types):
        for g in range(2):
            if x_polished[i, g] < tol:
                x_polished[i, g] = 0.0
            elif x_polished[i, g] > 1.0 - tol:
                x_polished[i, g] = 1.0
    
    # Check if polishing broke monotonicity or payment constraints
    # If so, revert to original
    if not _check_lp_constraints(x_polished, values, n_types):
        return x_alloc
    
    return x_polished

def _check_lp_constraints(x_alloc, values, n_types):
    """
    Checks monotonicity and non-negative payments for a given allocation.
    """
    # Monotonicity
    for v2 in range(4):
        for v1 in range(3):
            i_curr = v1 * 4 + v2
            i_next = (v1 + 1) * 4 + v2
            if x_alloc[i_curr, 0] > x_alloc[i_next, 0] + 1e-9:
                return False
            
    for v1 in range(4):
        for v2 in range(3):
            i_curr = v1 * 4 + v2
            i_next = v1 * 4 + (v2 + 1)
            if x_alloc[i_curr, 1] > x_alloc[i_next, 1] + 1e-9:
                return False
                
    # Non-negative payments
    payments = _calculate_payments(x_alloc, values, n_types)
    if np.any(payments < -1e-9):
        return False
        
    return True

def _calculate_payments(x_alloc, values, n_types):
    """
    Calculates payments using the IC identity.
    """
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
        
    return payments

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
    
    if np.any(allocs < -eps) or np.any(allocs > 1 + eps):
        return False
    if np.any(pays < -eps):
        return False
        
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
    """
    n_types = 16
    allocs = np.zeros((n_types, 2))
    pays = np.zeros(n_types)
    return {
        'allocations': allocs,
        'payments': pays
    }
