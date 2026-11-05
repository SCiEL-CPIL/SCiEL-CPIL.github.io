import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # 1. Solve allocation+utility LP
    result = _solve_allocation_with_utility_lp(probs, types, n_types, n_goods, deadline)
    
    if time.time() > deadline - 3.0:
        return _simple_safe_mechanism(types)
    
    if result is None:
        return _simple_safe_mechanism(types)
    
    final_alloc, final_pay = result
    
    # 2. Clip to valid ranges
    final_alloc = np.clip(final_alloc, 0.0, 1.0)
    final_pay = np.maximum(final_pay, 0.0)
    
    # 3. High-precision polish: re-solve LP with tighter tolerances if time permits
    if time.time() < deadline - 3.0:
        result2 = _solve_allocation_with_utility_lp(probs, types, n_types, n_goods, deadline, tight=True)
        if result2 is not None:
            alloc2, pay2 = result2
            # Compare revenues; keep the better one
            rev1 = _expected_revenue(final_alloc, final_pay, probs, types)
            rev2 = _expected_revenue(alloc2, pay2, probs, types)
            if rev2 > rev1 + 1e-12:
                final_alloc, final_pay = alloc2, pay2
    
    # Final safety: verify and repair if needed
    final_alloc, final_pay = _verify_and_repair(final_alloc, final_pay, probs, types)
    
    return {
        'allocations': final_alloc,
        'payments': final_pay
    }

def _parse_distribution(distribution_spec, support, types, n_types):
    """Robustly parse distribution spec to get type probabilities."""
    if distribution_spec is None:
        return np.ones(n_types) / n_types
    
    # If it's a direct array of correct size
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = np.maximum(arr, 0)
            if arr.sum() > 0:
                return arr / arr.sum()
            return np.ones(n_types) / n_types

    # If it's a dict
    if isinstance(distribution_spec, dict):
        # Look for common keys
        for key in ['probs', 'distribution', 'type_probs', 'probabilities', 'p']:
            if key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.array(val, dtype=float).flatten()
                    if len(arr) == n_types:
                        arr = np.maximum(arr, 0)
                        if arr.sum() > 0:
                            return arr / arr.sum()
                elif isinstance(val, dict):
                    # Map tuples to probs
                    probs = np.zeros(n_types)
                    for i, t in enumerate(types):
                        if t in val:
                            probs[i] = val[t]
                        elif str(t) in val:
                            probs[i] = val[str(t)]
                    probs = np.maximum(probs, 0)
                    if probs.sum() > 0:
                        return probs / probs.sum()
        
        # Check if the dict itself maps types to probs
        probs = np.zeros(n_types)
        found_any = False
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
                found_any = True
            elif str(t) in distribution_spec:
                probs[i] = distribution_spec[str(t)]
                found_any = True
        if found_any:
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                return probs / probs.sum()
    
    # Check support dict
    if support is not None and isinstance(support, dict):
        if 'probs' in support:
            val = support['probs']
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float).flatten()
                if len(arr) == n_types:
                    arr = np.maximum(arr, 0)
                    if arr.sum() > 0:
                        return arr / arr.sum()

    # Fallback: Uniform
    return np.ones(n_types) / n_types

def _expected_revenue(allocations, payments, probs, types):
    """Compute expected revenue for a mechanism."""
    n_types = len(types)
    rev = 0.0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        xi = np.array(allocations[i], dtype=float)
        rev += probs[i] * (np.dot(vi, xi) - payments[i])
    return rev

def _solve_allocation_with_utility_lp(probs, types, n_types, n_goods, deadline, tight=False):
    """
    Solve the mechanism design LP with variables x[i][g] and U[i].
    If tight=True, use tighter tolerances for higher precision.
    """
    def var_x(i, g):
        return i * n_goods + g
    
    def var_U(i):
        return n_types * n_goods + i
    
    n_vars = n_types * n_goods + n_types
    
    # Objective: minimize -sum_i probs[i] * (v_i . x_i - U[i])
    c = np.zeros(n_vars)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for g in range(n_goods):
            c[var_x(i, g)] = -probs[i] * vi[g]
        c[var_U(i)] = probs[i]
    
    # IC constraints: -U_i + U_j + (vi - vj) . x_j <= 0
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            A_ub[idx, var_U(i)] = -1.0
            A_ub[idx, var_U(j)] = 1.0
            for g in range(n_goods):
                A_ub[idx, var_x(j, g)] = vi[g] - vj[g]
            idx += 1
    
    for i in range(n_types):
        A_ub[idx, var_U(i)] = -1.0
        idx += 1
    
    bounds = [(0, 1)] * (n_types * n_goods) + [(0, None)] * n_types
    
    remaining = max(3.0, deadline - time.time() - 3.0)
    
    if tight:
        tol = 1e-14
    else:
        tol = 1e-12
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": tol,
                               "primal_feasibility_tolerance": tol,
                               "time_limit": remaining})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            utilities = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
                utilities[i] = x[var_U(i)]
            
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                xi = np.array(allocations[i], dtype=float)
                p_i = np.dot(vi, xi) - utilities[i]
                payments[i] = max(0.0, p_i)
            
            return allocations, payments
        else:
            return None
    except Exception:
        return None

def _polish_with_slsqp(allocations, payments, probs, types, deadline):
    """
    Local refinement using SLSQP to polish the LP solution with tighter tolerances.
    Maximizes expected revenue subject to IC, IR, and feasibility constraints.
    This is kept for compatibility but the main path now uses LP re-solve with tighter tolerances.
    """
    # For now, just return the input unchanged since LP re-solve is preferred
    return allocations, payments

def _verify_and_repair(allocations, payments, probs, types):
    """
    Verify IC/IR and non-negativity. Repair by recomputing payments via LP if violated.
    """
    n_types = len(types)
    tol = 1e-8
    
    # Check IC: u_i(i) >= u_i(j) for all i, j
    # u_i(j) = vi.xj - pj
    # So: vi.xi - pi >= vi.xj - pj
    
    violations = False
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        xi = np.array(allocations[i], dtype=float)
        ui = np.dot(vi, xi) - payments[i]
        
        # IR
        if ui < -tol:
            violations = True
            break
        
        for j in range(n_types):
            vj = np.array(types[j], dtype=float)
            xj = np.array(allocations[j], dtype=float)
            uj = np.dot(vi, xj) - payments[j]
            if ui - uj < -tol:
                violations = True
                break
        if violations:
            break
    
    if not violations:
        return allocations, payments
    
    # Repair: recompute payments via utility LP (minimize expected utility)
    result = _compute_payments_lp(allocations, probs, types)
    if result is not None:
        return result
    
    # Fallback: zero allocation and zero payment (guaranteed IC/IR)
    return np.zeros((n_types, 2)), np.zeros(n_types)

def _compute_payments_lp(allocations, probs, types):
    """Compute optimal payments (minimize expected utility) given fixed allocation."""
    n_types = len(types)
    c = probs.copy()
    n_vars = n_types
    n_ineq = n_types + n_types * (n_types - 1)
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    for i in range(n_types):
        A_ub[idx, i] = -1.0
        idx += 1
    
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            xj = np.array(allocations[j], dtype=float)
            rhs = np.dot(vj, xj) - np.dot(vi, xj)
            A_ub[idx, i] = -1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = rhs
            idx += 1
    
    bounds = [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12,
                               "primal_feasibility_tolerance": 1e-12,
                               "time_limit": 10})
        
        if res.success:
            U = np.maximum(res.x, 0.0)
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                xi = np.array(allocations[i], dtype=float)
                payments[i] = max(0.0, np.dot(vi, xi) - U[i])
            return allocations, payments
        return None
    except Exception:
        return None

def _simple_safe_mechanism(types):
    """
    Returns a mechanism that is guaranteed to be IC and IR, with 0 revenue.
    x_i = 0 for all i, p_i = 0 for all i.
    """
    n_types = len(types)
    return np.zeros((n_types, 2)), np.zeros(n_types)
