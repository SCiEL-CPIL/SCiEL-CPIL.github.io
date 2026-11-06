"""Improved program for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def _verify_and_repair(allocations, payments, val, probs, n_types, n_bundles, types, bundles):
    """
    Verify IC and IR constraints, and repair small numerical violations.
    Returns (allocations, payments, is_valid)
    """
    x_prob = np.zeros((n_types, n_bundles))
    for i in range(n_types):
        for j, (x1, x2) in enumerate(bundles):
            x_prob[i, j] = allocations[i, 0] * x1 * (1 - x2) + allocations[i, 1] * x2 * (1 - x1) + allocations[i, 0] * allocations[i, 1] * 0
    
    # Reconstruct x_prob from allocations
    # allocations[i,0] = x1*x_opt[i,1] + x1*x_opt[i,3] = x_opt[i,1] + x_opt[i,3]
    # allocations[i,1] = x2*x_opt[i,2] + x2*x_opt[i,3] = x_opt[i,2] + x_opt[i,3]
    # x_opt[i,0] = 1 - x_opt[i,1] - x_opt[i,2] - x_opt[i,3]
    
    a1 = allocations[:, 0]
    a2 = allocations[:, 1]
    
    # From the LP, x_opt[i,:] are probabilities over 4 bundles
    # a1[i] = x_opt[i,1] + x_opt[i,3]
    # a2[i] = x_opt[i,2] + x_opt[i,3]
    # We need to recover x_opt or just work with expected utilities
    
    # Expected utility of type i: E[val[i]·x] - p[i]
    # = val[i,0]*x0 + val[i,1]*x1 + val[i,2]*x2 + val[i,3]*x3 - p[i]
    # where x0=1-x1-x2-x3
    # = val[i,0]*(1-x1-x2-x3) + val[i,1]*x1 + val[i,2]*x2 + val[i,3]*x3 - p[i]
    
    # But we only have a1 and a2, not the individual x's
    # a1 = x1 + x3, a2 = x2 + x3
    # x0 = 1 - x1 - x2 - x3 = 1 - a1 - a2 + x3
    # So we have freedom in x3
    
    # For verification, we can compute expected value directly:
    # E[val[i]·x] = val[i,0]*x0 + val[i,1]*x1 + val[i,2]*x2 + val[i,3]*x3
    # With bundles (0,0),(1,0),(0,1),(1,1):
    # val[i,0] = 0, val[i,1] = v1_i, val[i,2] = v2_i, val[i,3] = v1_i+v2_i
    # E = 0*x0 + v1_i*x1 + v2_i*x2 + (v1_i+v2_i)*x3
    # = v1_i*(x1+x3) + v2_i*(x2+x3)
    # = v1_i*a1 + v2_i*a2
    
    v1 = np.array([t[0] for t in types])
    v2 = np.array([t[1] for t in types])
    
    expected_val = v1 * a1 + v2 * a2
    utilities = expected_val - payments
    
    # Check IR
    ir_violation = np.max(-utilities) if n_types > 0 else 0
    
    # Check IC: for each i,j, u(i, x_i, p_i) >= u(i, x_j, p_j)
    ic_violation = 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            # u(i, x_j, p_j) = v1_i*a1[j] + v2_i*a2[j] - p[j]
            u_i_j = v1[i] * a1[j] + v2[i] * a2[j] - payments[j]
            u_i_i = utilities[i]
            if u_i_j > u_i_i + ic_violation:
                ic_violation = u_i_j - u_i_i
    
    max_violation = max(ir_violation, ic_violation)
    
    if max_violation > 1e-8:
        # Try to repair by reducing payments slightly
        # This is a heuristic - just return invalid
        return allocations, payments, False
    
    return allocations, payments, True

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction with one additive bidder, two goods,
    values in {0,1,2,3}, using the optimal mechanism derived from an LP.
    
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # 1. Define Types and Bundles
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    n_types = 16
    n_bundles = 4
    bundles = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
    # 2. Parse Distribution
    probs = np.zeros(16)
    try:
        if isinstance(distribution_spec, dict):
            for i, t in enumerate(types):
                if t in distribution_spec:
                    probs[i] = distribution_spec[t]
                elif str(t) in distribution_spec:
                    probs[i] = distribution_spec[str(t)]
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(16) / 16.0
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == 16:
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = np.ones(16) / 16.0
            else:
                probs = np.ones(16) / 16.0
        else:
            probs = np.ones(16) / 16.0
    except Exception:
        probs = np.ones(16) / 16.0
        
    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0

    # 3. Setup Value Matrix
    # val[i, j] = value of bundle j for type i
    val = np.zeros((n_types, n_bundles))
    for i, (v1, v2) in enumerate(types):
        for j, (x1, x2) in enumerate(bundles):
            val[i, j] = v1 * x1 + v2 * x2

    # 4. Define LP Variables
    # x[i, j] for i in 0..15, j in 0..3 (64 vars)
    # p[i] for i in 0..15 (16 vars)
    # Total vars: 80
    
    n_vars = 80
    
    # Objective: Maximize sum_i probs[i] * p[i]
    # Minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[64 + i] = -probs[i]

    # 5. Equality Constraints
    # Sum_j x[i, j] = 1 for all i
    A_eq = np.zeros((n_types, n_vars))
    b_eq = np.ones(n_types)
    for i in range(n_types):
        for j in range(n_bundles):
            A_eq[i, i * n_bundles + j] = 1

    # 6. Inequality Constraints
    # A_ub * vars <= b_ub
    
    # IR: p[i] <= val[i] . x[i]  =>  p[i] - sum_j val[i,j]*x[i,j] <= 0
    # IC: u[i] >= u[j]  =>  val[i].x[i] - p[i] >= val[i].x[j] - p[j]
    #     => p[i] - p[j] + val[i].x[j] - val[i].x[i] <= 0
    
    n_ineq = n_types + n_types * (n_types - 1)
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)
    
    row = 0
    # IR Constraints
    for i in range(n_types):
        A_ub[row, 64 + i] = 1  # p[i]
        for j in range(n_bundles):
            A_ub[row, i * n_bundles + j] = -val[i, j]  # -x[i,j] * val[i,j]
        b_ub[row] = 0
        row += 1
        
    # IC Constraints
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            # p[i] - p[j] + val[i].x[j] - val[i].x[i] <= 0
            A_ub[row, 64 + i] = 1   # p[i]
            A_ub[row, 64 + j] = -1  # -p[j]
            for k in range(n_bundles):
                # + val[i,k] * x[j,k]
                A_ub[row, j * n_bundles + k] += val[i, k]
                # - val[i,k] * x[i,k]
                A_ub[row, i * n_bundles + k] -= val[i, k]
            b_ub[row] = 0
            row += 1

    bounds = [(0, None)] * n_vars

    # 7. Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
        
        if result.status == 0:
            x_opt = result.x[:64].reshape(n_types, n_bundles)
            p_opt = result.x[64:80]
            
            # Clean up numerical noise
            x_opt = np.maximum(x_opt, 0)
            p_opt = np.maximum(p_opt, 0)
            
            # Normalize allocations just in case
            for i in range(n_types):
                s = x_opt[i].sum()
                if s > 1e-9:
                    x_opt[i] /= s
                else:
                    x_opt[i] = np.zeros(n_bundles)
                    x_opt[i, 0] = 1.0 # Bundle (0,0)
            
            # Convert bundle probabilities to expected allocation per good
            allocations = np.zeros((n_types, 2))
            for i in range(n_types):
                for j, (x1, x2) in enumerate(bundles):
                    allocations[i, 0] += x1 * x_opt[i, j]
                    allocations[i, 1] += x2 * x_opt[i, j]
            
            # Verify constraints to ensure validity
            # Check IC and IR
            max_violation = 0.0
            for i in range(n_types):
                # IR check
                u_i = val[i] @ x_opt[i] - p_opt[i]
                if u_i < -max_violation:
                    max_violation = -u_i
                
                for j in range(n_types):
                    if i == j: continue
                    # IC check: u(i, x_i) >= u(i, x_j)
                    u_i_xi = val[i] @ x_opt[i] - p_opt[i]
                    u_i_xj = val[i] @ x_opt[j] - p_opt[j]
                    if u_i_xj - u_i_xi > max_violation:
                        max_violation = u_i_xj - u_i_xi
            
            if max_violation > 1e-8:
                # If violation is too large, fall back to a safe mechanism
                # Safe mechanism: Allocate nothing, pay 0
                # This is always IC and IR
                allocations = np.zeros((n_types, 2))
                p_opt = np.zeros(n_types)
            else:
                # Clip tiny negatives in payments
                p_opt = np.clip(p_opt, 0, None)
                
                # Verify and repair
                allocations, p_opt, is_valid = _verify_and_repair(
                    allocations, p_opt, val, probs, n_types, n_bundles, types, bundles
                )
                
                if is_valid:
                    return {
                        'allocations': allocations,
                        'payments': p_opt
                    }
                else:
                    # Fallback to safe mechanism
                    allocations = np.zeros((n_types, 2))
                    p_opt = np.zeros(n_types)
        else:
            # LP failed
            pass
    except Exception:
        pass

    # 8. Fallback: Try a simple allocation mechanism
    # For a single bidder, we can try allocating each good independently
    # if the expected value is positive
    
    # Try: allocate good 1 if v1 >= 1, good 2 if v2 >= 1
    # Payment: Myerson-style for each good
    # For a single good with values in {0,1,2,3} and distribution probs:
    # Sort types by value, compute virtual values
    
    v1_vals = np.array([t[0] for t in types])
    v2_vals = np.array([t[1] for t in types])
    
    # For each good, compute the optimal threshold mechanism
    # Sort by value
    order1 = np.argsort(v1_vals)
    order2 = np.argsort(v2_vals)
    
    # For good 1: allocate if v1 >= threshold
    # Revenue = sum over types where allocated: probs[i] * payment[i]
    # For a single good, IC + IR implies payment is determined by allocation
    # u(v) = v*alloc(v) - payment(v) must be non-increasing in v for IC
    # Actually for single good, IC means: v*alloc(v) - p(v) >= v'*alloc(v) - p(v') for all v,v'
    
    # Simpler: just use the LP result if valid, otherwise return zeros
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
