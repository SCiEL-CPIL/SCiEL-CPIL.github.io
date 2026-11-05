"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def _parse_distribution(distribution_spec, num_types, types):
    """Parse distribution spec into probability vector."""
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, int):
        probs = np.ones(num_types) / num_types
    else:
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == num_types:
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(num_types) / num_types
            else:
                probs = np.ones(num_types) / num_types
        except:
            probs = np.ones(num_types) / num_types
    return probs

def _build_lp_matrices(types, num_types):
    """
    Build LP constraint matrices for IC and IR.
    For additive valuations on a product lattice, IC is implied by 
    single-step monotonicity constraints (adjacent types in each dimension).
    """
    n_vars = 3 * num_types
    
    # Build list of constraints: (i, j) for IC where i dominates j in one step
    # For type (v1, v2), adjacent lower types are (v1-1, v2) and (v1, v2-1)
    # IC: u(i) >= u(i|j) where j is obtained by decreasing one coordinate by 1
    
    ic_pairs = []
    for v1 in range(4):
        for v2 in range(4):
            i = v1 * 4 + v2
            # Lower in v1
            if v1 > 0:
                j = (v1 - 1) * 4 + v2
                ic_pairs.append((i, j))
            # Lower in v2
            if v2 > 0:
                j = v1 * 4 + (v2 - 1)
                ic_pairs.append((i, j))
    
    n_ic = len(ic_pairs)
    n_ir = num_types
    n_constraints = n_ic + n_ir
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    row_idx = 0
    for i, j in ic_pairs:
        v1_i, v2_i = types[i]
        v1_j, v2_j = types[j]
        row = A_ub[row_idx]
        # IC: v1_i*x1[i] + v2_i*x2[i] - t[i] >= v1_i*x1[j] + v2_i*x2[j] - t[j]
        # => -v1_i*x1[i] - v2_i*x2[i] + t[i] + v1_i*x1[j] + v2_i*x2[j] - t[j] <= 0
        row[i] = -v1_i
        row[num_types + i] = -v2_i
        row[2 * num_types + i] = 1.0
        row[j] += v1_i
        row[num_types + j] += v2_i
        row[2 * num_types + j] -= 1.0
        row_idx += 1
    
    # IR constraints: t[i] <= v1_i*x1[i] + v2_i*x2[i]
    for i in range(num_types):
        v1_i, v2_i = types[i]
        row = A_ub[row_idx]
        row[i] = -v1_i
        row[num_types + i] = -v2_i
        row[2 * num_types + i] = 1.0
        row_idx += 1
    
    return A_ub, b_ub

def _verify_mechanism(allocations, payments, types, num_types, tol=1e-8):
    """Verify IC and IR constraints with vectorized computation."""
    x1 = allocations[:, 0]
    x2 = allocations[:, 1]
    t = payments
    
    # Vectorized IR check
    utils_self = np.array([types[i][0] * x1[i] + types[i][1] * x2[i] - t[i] for i in range(num_types)])
    if np.any(utils_self < -tol):
        return False
    
    # Vectorized IC check: for each i, check all j
    for i in range(num_types):
        v1_i, v2_i = types[i]
        u_ii = utils_self[i]
        # Compute u(i|j) for all j
        u_ij = v1_i * x1 + v2_i * x2 - t
        if np.any(u_ii < u_ij - tol):
            return False
    
    return True

def _fix_violations(allocations, payments, types, num_types):
    """Try to fix small IC/IR violations by adjusting payments with iterative refinement."""
    x1 = allocations[:, 0].copy()
    x2 = allocations[:, 1].copy()
    t = payments.copy()
    
    # Clip allocations
    x1 = np.clip(x1, 0, 1)
    x2 = np.clip(x2, 0, 1)
    t = np.clip(t, 0, None)
    
    # Ensure IR: t[i] <= v1_i*x1[i] + v2_i*x2[i]
    for i in range(num_types):
        v1_i, v2_i = types[i]
        max_t_ir = v1_i * x1[i] + v2_i * x2[i]
        if t[i] > max_t_ir:
            t[i] = max_t_ir
    
    # Iteratively fix IC violations
    for iteration in range(100):
        changed = False
        for i in range(num_types):
            v1_i, v2_i = types[i]
            u_ii_alloc = v1_i * x1[i] + v2_i * x2[i]
            # Find max deviation
            u_ij = v1_i * x1 + v2_i * x2 - t
            max_dev = np.max(u_ij)
            # t[i] must be <= u_ii_alloc - max_dev
            max_t_ic = u_ii_alloc - max_dev
            if t[i] > max_t_ic + 1e-12:
                t[i] = max(max_t_ic, 0.0)
                changed = True
        if not changed:
            break
    
    # Final clip to ensure non-negative
    t = np.clip(t, 0, None)
    
    return np.column_stack([x1, x2]), t

def _solve_lp(probs, types, num_types, A_ub, b_ub, bounds):
    """Solve the revenue-maximizing LP with high precision."""
    c = np.zeros(3 * num_types)
    for i in range(num_types):
        c[2 * num_types + i] = -probs[i]
    
    # Try HiGHS with tight tolerances
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                        options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                'primal_feasibility_tolerance': 1e-12})
        if result.success:
            return result.x
    except Exception:
        pass
    
    # Fallback to default HiGHS
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return result.x
    except Exception:
        pass
    
    # Try interior point method as backup
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
        if result.success:
            return result.x
    except Exception:
        pass
    
    return None

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder with two goods.
    
    Args:
        distribution_spec: dict mapping (v1, v2) to probability, or int index for standard distributions
        support: list of (v1, v2) pairs or similar
        constraints: dict with constraint parameters
        
    Returns:
        dict with 'allocations' (16, 2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    # Define the 16 types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)  # 16
    
    # Determine probabilities
    probs = _parse_distribution(distribution_spec, num_types, types)
    
    # Build LP matrices
    A_ub, b_ub = _build_lp_matrices(types, num_types)
    
    # Bounds
    bounds = []
    for i in range(num_types):
        bounds.append((0, 1))  # x1[i]
    for i in range(num_types):
        bounds.append((0, 1))  # x2[i]
    for i in range(num_types):
        bounds.append((0, None))  # t[i]
    
    best_result = None
    best_revenue = -np.inf
    
    # Solve the LP with HiGHS
    x_sol = _solve_lp(probs, types, num_types, A_ub, b_ub, bounds)
    
    if x_sol is not None:
        x1 = np.clip(x_sol[:num_types], 0, 1)
        x2 = np.clip(x_sol[num_types:2 * num_types], 0, 1)
        t = np.clip(x_sol[2 * num_types:3 * num_types], 0, None)
        
        allocations = np.column_stack([x1, x2])
        payments = t
        
        if _verify_mechanism(allocations, payments, types, num_types):
            revenue_val = np.sum(probs * t)
            best_result = {'allocations': allocations, 'payments': payments}
            best_revenue = revenue_val
            return best_result
        else:
            # Try to fix violations
            alloc_fixed, pay_fixed = _fix_violations(allocations, payments, types, num_types)
            if _verify_mechanism(alloc_fixed, pay_fixed, types, num_types):
                revenue_val = np.sum(probs * pay_fixed)
                best_result = {'allocations': alloc_fixed, 'payments': pay_fixed}
                best_revenue = revenue_val
                return best_result
    
    # Fallback: try a few heuristic mechanisms
    if time.time() < deadline:
        # Heuristic 1: proportional allocation
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x1 = np.full(num_types, alpha)
            x2 = np.full(num_types, alpha)
            t = np.array([types[i][0] * alpha + types[i][1] * alpha for i in range(num_types)])
            alloc = np.column_stack([x1, x2])
            if _verify_mechanism(alloc, t, types, num_types):
                rev = np.sum(probs * t)
                if rev > best_revenue:
                    best_result = {'allocations': alloc, 'payments': t}
                    best_revenue = rev
        
        # Heuristic 2: threshold mechanism
        for thresh in range(0, 8):
            x1 = np.array([1.0 if types[i][0] >= thresh else 0.0 for i in range(num_types)])
            x2 = np.array([1.0 if types[i][1] >= thresh else 0.0 for i in range(num_types)])
            # Compute payments to satisfy IC/IR
            t = np.zeros(num_types)
            for i in range(num_types):
                v1_i, v2_i = types[i]
                t[i] = v1_i * x1[i] + v2_i * x2[i]
            alloc = np.column_stack([x1, x2])
            if _verify_mechanism(alloc, t, types, num_types):
                rev = np.sum(probs * t)
                if rev > best_revenue:
                    best_result = {'allocations': alloc, 'payments': t}
                    best_revenue = rev
        
        # Heuristic 3: monotone allocation with optimized payments
        # Try different monotone allocation patterns
        for p1 in range(4):  # threshold for good 1
            for p2 in range(4):  # threshold for good 2
                x1 = np.array([1.0 if types[i][0] >= p1 else 0.0 for i in range(num_types)])
                x2 = np.array([1.0 if types[i][1] >= p2 else 0.0 for i in range(num_types)])
                # Compute optimal payments using LP for fixed allocations
                # t[i] should be as high as possible while maintaining IC/IR
                # This is equivalent to: t[i] = v1_i*x1[i] + v2_i*x2[i] - max_j(v1_i*x1[j] + v2_i*x2[j] - t[j])
                # With monotone allocations, we can compute payments greedily
                t = np.zeros(num_types)
                # Sort types by value (ascending) and compute payments
                order = sorted(range(num_types), key=lambda i: (types[i][0] + types[i][1], types[i][0]))
                for idx in order:
                    i = idx
                    v1_i, v2_i = types[i]
                    u_ii_alloc = v1_i * x1[i] + v2_i * x2[i]
                    # Max deviation from already-set payments
                    max_dev = 0.0
                    for j in range(num_types):
                        if t[j] > 0 or j == i:  # Only consider types with set payments or self
                            u_ij = v1_i * x1[j] + v2_i * x2[j] - t[j]
                            if u_ij > max_dev:
                                max_dev = u_ij
                    t[i] = max(u_ii_alloc - max_dev, 0.0)
                
                alloc = np.column_stack([x1, x2])
                if _verify_mechanism(alloc, t, types, num_types):
                    rev = np.sum(probs * t)
                    if rev > best_revenue:
                        best_result = {'allocations': alloc, 'payments': t}
                        best_revenue = rev
    
    if best_result is not None:
        return best_result
    
    # Final fallback: zero mechanism
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
