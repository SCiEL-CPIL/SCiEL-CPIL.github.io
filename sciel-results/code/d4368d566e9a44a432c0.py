import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.optimize import OptimizeResult
from scipy.sparse import csc_matrix

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
    
    # Build and solve LP first for a good initial point
    lp_result = _solve_lp(probs, types, n_types, n_goods)
    
    if time.time() > deadline - 5.0:
        # Fallback to zero mechanism if running out of time
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
    
    best_result = None
    if lp_result is not None:
        best_result = lp_result
        # Use LP solution as warm start for SLSQP refinement
        slsqp_result = _solve_slsqp(probs, types, n_types, n_goods, lp_result)
        if slsqp_result is not None:
            # Compare revenues
            rev_lp = _compute_revenue(lp_result, probs)
            rev_slsqp = _compute_revenue(slsqp_result, probs)
            if rev_slsqp >= rev_lp - 1e-10:
                best_result = slsqp_result
            else:
                best_result = lp_result
        # Try high-precision polish
        if time.time() < deadline - 10.0:
            polish_result = _polish_lp(probs, types, n_types, n_goods, best_result)
            if polish_result is not None:
                rev_polish = _compute_revenue(polish_result, probs)
                rev_best = _compute_revenue(best_result, probs)
                if rev_polish >= rev_best - 1e-10:
                    best_result = polish_result
        # Final SLSQP polish from best result for maximum precision
        if time.time() < deadline - 5.0:
            final_slsqp = _solve_slsqp(probs, types, n_types, n_goods, best_result)
            if final_slsqp is not None:
                rev_final = _compute_revenue(final_slsqp, probs)
                rev_best = _compute_revenue(best_result, probs)
                if rev_final >= rev_best - 1e-10:
                    best_result = final_slsqp
    else:
        # LP failed, try SLSQP from scratch
        slsqp_result = _solve_slsqp(probs, types, n_types, n_goods, None)
        if slsqp_result is not None:
            best_result = slsqp_result
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to zero mechanism
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
        
        # Maybe it's a 4x4 matrix?
        if len(arr) == 16:
             # Already handled above
             pass
             
        # Maybe it's product of marginals? E.g., [p0, p1, p2, p3] for each good?
        # This is a guess. If length is 4, assume independent marginals.
        if len(arr) == 4:
            # arr[i] is prob for value i?
            # This is ambiguous. Let's stick to uniform if not 16.
            pass
        
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
        # Use HiGHS solver, which is robust and fast for this size
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
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

def _solve_slsqp(probs, types, n_types, n_goods, warm_start=None):
    """
    Solve the revenue-maximizing IC/IR problem using SLSQP with explicit
    differentiable constraints. Uses a pre-computed constant constraint Jacobian
    for efficiency since all constraints are linear.
    
    Variables: z = [x_0,0, x_0,1, ..., x_15,0, x_15,1, p_0, ..., p_15]
    Total: 2*n_types + n_types = 3*n_types = 48 variables
    
    Objective: Minimize -sum_i probs[i] * p[i]  (maximize revenue)
    
    Constraints (all <= 0):
    1. IC: for all i != j: v_i . x_j - p_j - v_i . x_i + p_i <= 0
    2. IR: for all i: -v_i . x_i + p_i <= 0
    """
    n_vars = 3 * n_types
    
    def var_x_idx(i, g):
        return 2 * i + g
    
    def var_p_idx(i):
        return 2 * n_types + i
    
    # Precompute value vectors
    V = np.array(types, dtype=float)  # shape (n_types, 2)
    
    # Number of constraints
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_constraints = n_ic + n_ir
    
    # Precompute the constant constraint Jacobian matrix A (n_constraints x n_vars)
    # For IC constraint (i, j): v_i . x_j - p_j - v_i . x_i + p_i <= 0
    # For IR constraint i: -v_i . x_i + p_i <= 0
    A = np.zeros((n_constraints, n_vars))
    
    ic_idx = []
    idx = 0
    for i in range(n_types):
        vi = V[i]
        for j in range(n_types):
            if i == j:
                continue
            ic_idx.append((i, j))
            # d/dx_j: +vi
            A[idx, var_x_idx(j, 0)] = vi[0]
            A[idx, var_x_idx(j, 1)] = vi[1]
            # d/dp_j: -1
            A[idx, var_p_idx(j)] = -1.0
            # d/dx_i: -vi
            A[idx, var_x_idx(i, 0)] = -vi[0]
            A[idx, var_x_idx(i, 1)] = -vi[1]
            # d/dp_i: +1
            A[idx, var_p_idx(i)] = 1.0
            idx += 1
    
    # IR constraints
    for i in range(n_types):
        vi = V[i]
        row = n_ic + i
        # d/dx_i: -vi
        A[row, var_x_idx(i, 0)] = -vi[0]
        A[row, var_x_idx(i, 1)] = -vi[1]
        # d/dp_i: +1
        A[row, var_p_idx(i)] = 1.0
    
    # Objective: minimize -expected revenue
    def objective(z):
        payments = z[2*n_types:]
        return -np.dot(probs, payments)
    
    # Objective gradient (constant)
    obj_grad = np.zeros(n_vars)
    obj_grad[2*n_types:] = -probs
    
    # Constraints: A @ z <= 0, so inequality fun returns -A @ z >= 0
    def constraint_fun(z):
        return -A @ z
    
    # Constraint Jacobian (constant, negative of A since we return -A@z)
    def constraint_jac(z):
        return -A
    
    # Bounds: x in [0,1], p in [0, inf)
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
    
    # Initial point
    if warm_start is not None:
        z0 = np.zeros(n_vars)
        for i in range(n_types):
            for g in range(n_goods):
                z0[var_x_idx(i, g)] = warm_start['allocations'][i, g]
            z0[var_p_idx(i)] = warm_start['payments'][i]
    else:
        # Start from a simple feasible point: zero allocations, zero payments
        z0 = np.zeros(n_vars)
    
    # Use tighter tolerances and more iterations for high precision
    try:
        res = minimize(
            objective,
            z0,
            jac=obj_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': constraint_fun, 'jac': constraint_jac}
            ],
            options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False}
        )
        
        if res.success:
            z = res.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = float(np.clip(z[var_x_idx(i, g)], 0, 1))
                payments[i] = float(max(0, z[var_p_idx(i)]))
            
            # Final verification
            allocations, payments = _verify_and_fix(allocations, payments, types)
            
            # Verify feasibility more strictly
            if _check_feasibility(allocations, payments, types, tol=1e-8):
                return {'allocations': allocations, 'payments': payments}
            else:
                # Try to fix violations
                allocations, payments = _fix_violations(allocations, payments, types)
                if _check_feasibility(allocations, payments, types, tol=1e-8):
                    return {'allocations': allocations, 'payments': payments}
                else:
                    return None
        else:
            return None
    except Exception:
        return None

def _compute_revenue(result, probs):
    """Compute expected revenue from a result dictionary."""
    if result is None:
        return 0.0
    allocations = result['allocations']
    payments = result['payments']
    return float(np.dot(probs, payments))

def _polish_lp(probs, types, n_types, n_goods, warm_start):
    """
    High-precision LP polish using tight tolerances.
    Uses the warm start to guide the solution and applies tighter constraints.
    """
    def var_x(i, g):
        return 2 * i + g
    
    def var_p(i):
        return 2 * n_types + i
    
    n_vars = 3 * n_types
    
    # Objective: Maximize sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # Build constraints with tighter tolerance
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
            # IC: -vi.x_i + p_i + vi.x_j - p_j <= 0
            A_ub[idx, var_x(i, 0)] = -vi[0]
            A_ub[idx, var_x(i, 1)] = -vi[1]
            A_ub[idx, var_p(i)] = 1.0
            A_ub[idx, var_x(j, 0)] = vi[0]
            A_ub[idx, var_x(j, 1)] = vi[1]
            A_ub[idx, var_p(j)] = -1.0
            b_ub[idx] = 0.0
            idx += 1
            
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        # IR: -vi.x_i + p_i <= 0
        A_ub[idx, var_x(i, 0)] = -vi[0]
        A_ub[idx, var_x(i, 1)] = -vi[1]
        A_ub[idx, var_p(i)] = 1.0
        b_ub[idx] = 0.0
        idx += 1
        
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    try:
        # Use HiGHS with tight tolerances for high precision
        res = linprog(
            c, 
            A_ub=A_ub, 
            b_ub=b_ub, 
            bounds=bounds, 
            method='highs',
            options={
                'primal_feasibility_tolerance': 1e-14,
                'dual_feasibility_tolerance': 1e-14,
                'presolve': 'on'
            }
        )
        
        if res.success:
            x = res.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            
            for i in range(n_types):
                for g in range(n_goods):
                    allocations[i, g] = float(np.clip(x[var_x(i, g)], 0, 1))
                payments[i] = float(max(0, x[var_p(i)]))
            
            allocations, payments = _verify_and_fix(allocations, payments, types)
            
            # Verify this is actually better or equal
            rev_new = float(np.dot(probs, payments))
            rev_old = _compute_revenue(warm_start, probs)
            
            if rev_new >= rev_old - 1e-10:
                return {'allocations': allocations, 'payments': payments}
            else:
                return None
        else:
            return None
    except Exception:
        return None

def _check_feasibility(allocations, payments, types, tol=1e-8):
    """Check if allocations and payments satisfy IC and IR constraints."""
    n_types = len(types)
    
    # Check IR
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        if ui < -tol:
            return False
    
    # Check IC
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui = np.dot(vi, allocations[i]) - payments[i]
        for j in range(n_types):
            if i == j:
                continue
            uj_cross = np.dot(vi, allocations[j]) - payments[j]
            if ui < uj_cross - tol:
                return False
    
    # Check bounds
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False
    if np.any(payments < -tol):
        return False
    
    return True

def _fix_violations(allocations, payments, types):
    """Fix small violations in IC/IR constraints."""
    n_types = len(types)
    allocations = np.array(allocations, dtype=float)
    payments = np.array(payments, dtype=float)
    
    # Iteratively fix violations
    for iteration in range(10):
        # Fix IR violations
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            ui = np.dot(vi, allocations[i]) - payments[i]
            if ui < -1e-10:
                payments[i] = max(0.0, np.dot(vi, allocations[i]))
        
        # Fix IC violations by adjusting payments
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            ui = np.dot(vi, allocations[i]) - payments[i]
            for j in range(n_types):
                if i == j:
                    continue
                uj_cross = np.dot(vi, allocations[j]) - payments[j]
                if ui < uj_cross - 1e-10:
                    # Increase payment for type i to make IC hold
                    # New payment: p_i <= v_i.x_i - v_i.x_j + p_j
                    max_p_i = np.dot(vi, allocations[i]) - np.dot(vi, allocations[j]) + payments[j]
                    if max_p_i > payments[i]:
                        payments[i] = min(max_p_i, np.dot(vi, allocations[i]))  # Also respect IR
        
        # Check if fixed
        if _check_feasibility(allocations, payments, types, tol=1e-8):
            break
    
    # Final clip
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    return allocations, payments

def _verify_and_fix(allocations, payments, types):
    """
    Verify IC/IR constraints and fix small numerical violations.
    """
    allocations, payments = _fix_violations(allocations, payments, types)
    return allocations, payments
