import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # 1. Solve allocation-only LP with utility variables
    result = _solve_allocation_with_utility_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 5.0:
        return _simple_safe_mechanism(types)
    
    if result is None:
        return _simple_safe_mechanism(types)
    
    final_alloc, final_pay = result
    
    # 2. Final safety check and clipping
    final_alloc = np.clip(final_alloc, 0.0, 1.0)
    final_pay = np.maximum(final_pay, 0.0)
    
    # 3. Polish pass: verify constraints and repair if needed
    final_alloc, final_pay = _polish_mechanism(final_alloc, final_pay, probs, types)
    
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

def _solve_allocation_with_utility_lp(probs, types, n_types, n_goods):
    """
    Solve the mechanism design LP with variables x[i][g] and U[i].
    
    Variables:
    - x[i][g] for i in range(n_types), g in range(n_goods): allocation probabilities
    - U[i] for i in range(n_types): utility of type i
    
    Objective: maximize sum_i probs[i] * (v_i . x_i - U[i])
    This equals expected revenue since p_i = v_i . x_i - U_i.
    
    Constraints:
    1. IC: U_i - U_j >= v_i . x_j - v_j . x_j for all i, j
       Equivalent to: -U_i + U_j <= v_j . x_j - v_i . x_j
    2. IR: U_i >= 0
    3. 0 <= x[i][g] <= 1
    """
    # Variable indexing
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
        c[var_U(i)] = probs[i]  # minimize -(... - U_i) => coefficient of U_i is +probs[i]
    
    # Build inequality constraints A_ub @ var <= b_ub
    # IC: -U_i + U_j + v_i . x_j - v_j . x_j <= 0 for all i != j
    #    => -U_i + U_j + sum_g (vi[g] - vj[g]) * x_j[g] <= 0
    # IR: -U_i <= 0
    
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
            # -U_i + U_j + (vi - vj) . x_j <= 0
            A_ub[idx, var_U(i)] = -1.0
            A_ub[idx, var_U(j)] = 1.0
            for g in range(n_goods):
                A_ub[idx, var_x(j, g)] = vi[g] - vj[g]
            idx += 1
    
    for i in range(n_types):
        A_ub[idx, var_U(i)] = -1.0
        idx += 1
    
    bounds = [(0, 1)] * (n_types * n_goods) + [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-12,
                               "primal_feasibility_tolerance": 1e-12,
                               "time_limit": 35})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            utilities = np.zeros(n_types)
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = x[var_x(i, g)]
                utilities[i] = x[var_U(i)]
            
            # Compute payments using analytic utility computation for precision
            allocations, payments = _compute_optimal_payments(allocations, probs, types)
            
            if payments is None:
                # Fallback to LP-computed utilities
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

def _polish_mechanism(allocations, payments, probs, types):
    """
    Verify IC/IR constraints and repair violations using a conservative approach.
    """
    n_types = len(types)
    tol = 1e-10
    
    # First, clip allocations to valid range
    allocations = np.clip(allocations, 0.0, 1.0)
    
    # Recompute payments from the allocation using optimal utility computation
    result = _compute_optimal_payments(allocations, probs, types)
    if result is not None:
        return result
    
    # Fallback: use zero mechanism
    return np.zeros((n_types, 2)), np.zeros(n_types)

def _compute_optimal_payments(allocations, probs, types):
    """
    Given an allocation x, compute optimal payments using the utility LP.
    Minimize sum_i probs[i] * U_i subject to IC and IR.
    """
    n_types = len(types)
    
    # Objective: minimize sum probs[i] * U_i
    c = probs.copy()
    
    # Constraints:
    # 1. -U_i <= 0  (U_i >= 0)
    # 2. -U_i + U_j <= v_j . x_j - v_i . x_j
    
    n_vars = n_types
    n_ineq = n_types + n_types * (n_types - 1)
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    # IR: -U_i <= 0
    for i in range(n_types):
        A_ub[idx, i] = -1.0
        b_ub[idx] = 0.0
        idx += 1
    
    # IC: -U_i + U_j <= v_j . x_j - v_i . x_j
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            xj = np.array(allocations[j], dtype=float)
            # RHS: v_j . x_j - v_i . x_j
            rhs = np.dot(vj, xj) - np.dot(vi, xj)
            
            A_ub[idx, i] = -1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = rhs
            idx += 1
            
    bounds = [(0, None)] * n_types
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={"presolve": "on", "dual_feasibility_tolerance": 1e-14,
                               "primal_feasibility_tolerance": 1e-14,
                               "time_limit": 15})
        
        if res.success:
            U = res.x
            U = np.maximum(U, 0.0)
            
            # Verify IC constraints are satisfied
            max_violation = 0.0
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                for j in range(n_types):
                    if i == j:
                        continue
                    vj = np.array(types[j], dtype=float)
                    xj = np.array(allocations[j], dtype=float)
                    # IC: U_i - U_j >= v_i . x_j - v_j . x_j
                    lhs = U[i] - U[j]
                    rhs = np.dot(vi, xj) - np.dot(vj, xj)
                    violation = rhs - lhs
                    if violation > max_violation:
                        max_violation = violation
            
            if max_violation > 1e-8:
                # IC violated, try to repair by increasing utilities
                U = _repair_utilities(U, allocations, types, n_types)
                if U is None:
                    return None
            
            payments = np.zeros(n_types)
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                xi = np.array(allocations[i], dtype=float)
                p_i = np.dot(vi, xi) - U[i]
                payments[i] = max(0.0, p_i)
            
            # Final verification
            if _verify_mechanism(allocations, payments, probs, types):
                return allocations, payments
            else:
                return None
        else:
            return None
            
    except Exception:
        return None

def _repair_utilities(U, allocations, types, n_types):
    """
    Repair utilities to satisfy IC by computing the minimal feasible utility vector.
    Uses the fact that U_i must be at least max over all j of (U_j + v_i.x_j - v_j.x_j)
    and at least 0.
    """
    # Iterative fixed-point: U_i = max(0, max_j (U_j + v_i.x_j - v_j.x_j))
    U_new = U.copy()
    for iteration in range(100):
        U_old = U_new.copy()
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            min_u = 0.0
            for j in range(n_types):
                if i == j:
                    continue
                vj = np.array(types[j], dtype=float)
                xj = np.array(allocations[j], dtype=float)
                # U_i >= U_j + v_i.x_j - v_j.x_j
                required = U_old[j] + np.dot(vi, xj) - np.dot(vj, xj)
                if required > min_u:
                    min_u = required
            U_new[i] = max(0.0, min_u)
        
        if np.max(np.abs(U_new - U_old)) < 1e-14:
            break
    
    # Verify
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            xj = np.array(allocations[j], dtype=float)
            if U_new[i] - U_new[j] < np.dot(vi, xj) - np.dot(vj, xj) - 1e-8:
                return None
    
    return U_new

def _verify_mechanism(allocations, payments, probs, types):
    """Verify all constraints: feasibility, IR, IC, non-negative payments."""
    n_types = len(types)
    
    # Check allocation bounds
    if np.any(allocations < -1e-10) or np.any(allocations > 1 + 1e-10):
        return False
    
    # Check non-negative payments
    if np.any(payments < -1e-10):
        return False
    
    # Compute utilities
    U = np.zeros(n_types)
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        xi = np.array(allocations[i], dtype=float)
        U[i] = np.dot(vi, xi) - payments[i]
    
    # Check IR: U_i >= 0
    if np.any(U < -1e-8):
        return False
    
    # Check IC: U_i - U_j >= v_i.x_j - v_j.x_j for all i, j
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = np.array(types[j], dtype=float)
            xj = np.array(allocations[j], dtype=float)
            if U[i] - U[j] < np.dot(vi, xj) - np.dot(vj, xj) - 1e-8:
                return False
    
    return True

def _simple_safe_mechanism(types):
    """
    Returns a mechanism that is guaranteed to be IC and IR, with 0 revenue.
    x_i = 0 for all i, p_i = 0 for all i.
    """
    n_types = len(types)
    return np.zeros((n_types, 2)), np.zeros(n_types)
