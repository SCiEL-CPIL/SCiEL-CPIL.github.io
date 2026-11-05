import time
import numpy as np
from scipy.optimize import linprog, minimize

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible and individually rational auction mechanism.
    
    Args:
        distribution_spec: Specification of the probability distribution over types.
        support: The support of the types (usually {(0,0), ..., (3,3)}).
        constraints: Additional constraints (not explicitly detailed, assumed standard).
        
    Returns:
        A dictionary with 'allocations' (16, 2) and 'payments' (16,).
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, support, types, n_types)
    
    # Build and solve LP
    result = _solve_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 5.0:
        # Fallback to zero mechanism if running out of time
        if result is None:
            return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
        return result
        
    if result is not None:
        # Attempt to polish with SLSQP if time permits
        if time.time() < deadline - 2.0:
            polished = _solve_slsqp(result, probs, types, n_types, n_goods, start_time, deadline)
            if polished is not None:
                return polished
        return result
    else:
        # Fallback to zero mechanism if LP fails
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}

def _parse_distribution(distribution_spec, support, types, n_types):
    """
    Parse the distribution spec into an array of probabilities for the 16 types.
    Order: (0,0), (0,1), ..., (0,3), (1,0), ..., (3,3).
    """
    uniform = np.ones(n_types) / n_types
    
    if distribution_spec is None:
        return uniform
    
    # If it's a dict mapping types to probs
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n_types)
        found_any = False
        for i, t in enumerate(types):
            # Try tuple key
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
                found_any = True
            # Try string key "(v1, v2)"
            elif f"({t[0]}, {t[1]})" in distribution_spec:
                probs[i] = distribution_spec[f"({t[0]}, {t[1]})"]
                found_any = True
            # Try string key "v1,v2"
            elif f"{t[0]},{t[1]}" in distribution_spec:
                probs[i] = distribution_spec[f"{t[0]},{t[1]}"]
                found_any = True
            # Try flattened string key "v1v2"
            elif f"{t[0]}{t[1]}" in distribution_spec:
                probs[i] = distribution_spec[f"{t[0]}{t[1]}"]
                found_any = True
        
        if found_any:
            probs = np.maximum(probs, 0)
            if probs.sum() > 0:
                return probs / probs.sum()
            return uniform
        return uniform
    
    # If it's a list or array
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            arr = np.maximum(arr, 0)
            if arr.sum() > 0:
                return arr / arr.sum()
        
        # Maybe it's product of marginals? E.g., [p0, p1, p2, p3] for each good?
        if len(arr) == 4:
            # Assume independent marginals
            p1 = np.array([arr[0], arr[1], arr[2], arr[3]], dtype=float)
            p1 = np.maximum(p1, 0)
            if p1.sum() > 0:
                p1 = p1 / p1.sum()
                probs = np.outer(p1, p1)
                probs = probs.flatten()
                return probs
        
        return uniform
    
    # If it's a string, might be a name
    if isinstance(distribution_spec, str):
        if distribution_spec.lower() in ['uniform', 'unif']:
            return uniform
        # Try to parse JSON?
        try:
            import json
            spec = json.loads(distribution_spec)
            return _parse_distribution(spec, support, types, n_types)
        except:
            pass
        return uniform

    # Fallback: Check support
    if support is not None and isinstance(support, dict):
        if 'probs' in support:
            val = support['probs']
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float).flatten()
                if len(arr) == n_types:
                    arr = np.maximum(arr, 0)
                    if arr.sum() > 0:
                        return arr / arr.sum()

    return uniform

def _solve_lp(probs, types, n_types, n_goods):
    """
    Solve the revenue-maximizing IC/IR LP.
    Variables: x[0..2*n_types-1] for allocations, x[2*n_types..3*n_types-1] for payments.
    """
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 3 * n_types
    
    # Objective: Maximize sum_i probs[i] * p[i]
    # linprog minimizes, so we use negative.
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # Constraints:
    # 1. IC: for all i != j, v_i . x_i - p_i >= v_i . x_j - p_j
    #    => -v_i . x_i + p_i + v_i . x_j - p_j <= 0
    # 2. IR: for all i, v_i . x_i - p_i >= 0
    #    => -v_i . x_i + p_i <= 0
    
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_ineq = n_ic + n_ir
    
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    idx = 0
    # IC Constraints
    for i in range(n_types):
        vi = types[i]
        vi_vec = np.array(vi, dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            vj = types[j]
            # Row: -vi[0]*x_i0 - vi[1]*x_i1 + p_i + vi[0]*x_j0 + vi[1]*x_j1 - p_j <= 0
            A_ub[idx, var_x(i, 0)] = -vi_vec[0]
            A_ub[idx, var_x(i, 1)] = -vi_vec[1]
            A_ub[idx, var_p(i)] = 1.0
            A_ub[idx, var_x(j, 0)] = vi_vec[0]
            A_ub[idx, var_x(j, 1)] = vi_vec[1]
            A_ub[idx, var_p(j)] = -1.0
            b_ub[idx] = 0.0
            idx += 1
            
    # IR Constraints
    for i in range(n_types):
        vi = types[i]
        vi_vec = np.array(vi, dtype=float)
        # Row: -vi[0]*x_i0 - vi[1]*x_i1 + p_i <= 0
        A_ub[idx, var_x(i, 0)] = -vi_vec[0]
        A_ub[idx, var_x(i, 1)] = -vi_vec[1]
        A_ub[idx, var_p(i)] = 1.0
        b_ub[idx] = 0.0
        idx += 1
        
    # Bounds:
    # Allocations x[i,g] in [0, 1]
    # Payments p[i] in [0, infinity)
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    try:
        # Use HiGHS solver with tight tolerances for better precision
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True,
                              'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12,
                              'dual_objective_bound_tolerance': 1e-12})
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            
            for i in range(n_types):
                for g in range(n_goods):
                    # Clip to ensure feasibility
                    allocations[i, g] = float(np.clip(x[var_x(i, g)], 0, 1))
                # Ensure non-negative payment
                payments[i] = float(max(0, x[var_p(i)]))
            
            # Final verification and tiny fix for numerical drift
            allocations, payments = _verify_and_fix(allocations, payments, types)
            
            return {'allocations': allocations, 'payments': payments}
        else:
            return None
    except Exception:
        return None

def _solve_slsqp(init_result, probs, types, n_types, n_goods, start_time, deadline):
    """
    Polish the LP solution using SLSQP to ensure strict feasibility and maximize revenue.
    """
    def flatten(allocations, payments):
        return np.concatenate([allocations.flatten(), payments])

    def unflatten(vec):
        n_alloc = n_types * n_goods
        allocations = vec[:n_alloc].reshape(n_types, n_goods)
        payments = vec[n_alloc:]
        return allocations, payments

    def objective(vec):
        allocations, payments = unflatten(vec)
        return -np.sum(probs * payments)

    def jac_objective(vec):
        # Gradient of -sum(p_i * p_i) w.r.t vec
        # Only payments part has gradient -probs
        n_alloc = n_types * n_goods
        grad = np.zeros_like(vec)
        grad[n_alloc:] = -probs
        return grad

    def constraints(vec):
        allocations, payments = unflatten(vec)
        cons = []
        # IC: v_i . x_i - p_i - (v_i . x_j - p_j) >= 0
        # => p_j - v_i . x_j + v_i . x_i - p_i <= 0
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            for j in range(n_types):
                if i == j:
                    continue
                cons.append({
                    'type': 'ineq',
                    'fun': lambda a=allocations, p=payments, i=i, j=j: (np.dot(vi, a[j]) - p[j]) - (np.dot(vi, a[i]) - p[i]),
                    'jac': lambda a=allocations, p=payments, i=i, j=j: _jac_ic(a, p, i, j, n_types, n_goods)
                })
        # IR: v_i . x_i - p_i >= 0
        # => p_i - v_i . x_i <= 0
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            cons.append({
                'type': 'ineq',
                'fun': lambda a=allocations, p=payments, i=i: p[i] - np.dot(vi, a[i]),
                'jac': lambda a=allocations, p=payments, i=i: _jac_ir(a, p, i, n_types, n_goods)
            })
        return cons

    def _jac_ic(a, p, i, j, n_types, n_goods):
        # f = (v_i . a_j - p_j) - (v_i . a_i - p_i)
        # df/d(a_k, g)
        # if k == j: v_i[g]
        # if k == i: -v_i[g]
        # else 0
        # df/d(p_j) = -1
        # df/d(p_i) = 1
        # else 0
        n_alloc = n_types * n_goods
        grad = np.zeros(n_types * n_goods + n_types)
        vi = np.array(types[i], dtype=float)
        
        for g in range(n_goods):
            grad[j * n_goods + g] = vi[g]
            grad[i * n_goods + g] = -vi[g]
        
        grad[n_alloc + j] = -1.0
        grad[n_alloc + i] = 1.0
        return grad

    def _jac_ir(a, p, i, n_types, n_goods):
        # f = p_i - v_i . a_i
        # df/d(a_i, g) = -v_i[g]
        # df/d(p_i) = 1
        n_alloc = n_types * n_goods
        grad = np.zeros(n_types * n_goods + n_types)
        vi = np.array(types[i], dtype=float)
        
        for g in range(n_goods):
            grad[i * n_goods + g] = -vi[g]
        
        grad[n_alloc + i] = 1.0
        return grad

    try:
        x0 = flatten(init_result['allocations'], init_result['payments'])
        
        bounds = []
        for i in range(n_types):
            for g in range(n_goods):
                bounds.append((0, 1))
        for i in range(n_types):
            bounds.append((0, None))

        res = minimize(objective, x0, jac=jac_objective, method='SLSQP', 
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 100, 'ftol': 1e-12})
        
        if res.success and time.time() < deadline - 1.0:
            allocations, payments = unflatten(res.x)
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)
            
            # Verify and fix any residual issues
            allocations, payments = _verify_and_fix(allocations, payments, types)
            
            # Check if this is better than LP
            rev_lp = np.sum(probs * init_result['payments'])
            rev_slsqp = np.sum(probs * payments)
            
            if rev_slsqp >= rev_lp - 1e-9:
                return {'allocations': allocations, 'payments': payments}
        
        return None
    except Exception:
        return None

def _verify_and_fix(allocations, payments, types):
    """
    Verify IC/IR constraints and fix small numerical violations.
    """
    n_types = len(types)
    
    # Check IR and fix
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -1e-10:
            # Payment is too high, reduce it to satisfy IR
            payments[i] = max(0, np.dot(vi, allocations[i]))
            
    # Check IC and fix
    # If IC is violated, try to fix by adjusting payments or allocations
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        for j in range(n_types):
            if i == j:
                continue
            uj_cross = np.dot(vi, allocations[j]) - payments[j]
            if ui < uj_cross - 1e-8:
                # Violation found - increase payment for type i to reduce its utility
                needed_increase = uj_cross - ui + 1e-9
                payments[i] += needed_increase
                # Ensure IR is still satisfied
                if payments[i] > np.dot(vi, allocations[i]):
                    # Scale down allocation for type i
                    if np.dot(vi, allocations[i]) > 1e-12:
                        scale = (uj_cross + 1e-9) / np.dot(vi, allocations[i])
                        if scale < 1:
                            allocations[i] *= scale
                    else:
                        allocations[i] = 0
                    payments[i] = max(0, np.dot(vi, allocations[i]) - max(0, uj_cross - 1e-9))

    # Final clip
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # One more pass to ensure IR feasibility
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -1e-8:
            payments[i] = max(0, np.dot(vi, allocations[i]))
    
    return allocations, payments
