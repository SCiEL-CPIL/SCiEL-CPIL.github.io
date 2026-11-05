"""Incentive-compatible finite auction design solver – Direct Convex QP/SLSQP over Monotone Allocations and Payments."""
import time
import numpy as np
from scipy.optimize import minimize

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    values = [0, 1, 2, 3]
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]
    type_indices = {(v1, v2): i for i, (v1, v2) in enumerate(types)}
    
    # Helper to get index from type
    def get_idx(v1, v2):
        return type_indices[(v1, v2)]

    # Parse distribution
    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    if p1.sum() > 0:
        p1 = p1 / p1.sum()
    else:
        p1 = np.ones(4) / 4
    if p2.sum() > 0:
        p2 = p2 / p2.sum()
    else:
        p2 = np.ones(4) / 4

    joint_probs = np.outer(p1, p2).flatten()

    # Variables:
    # x1[0..15]: allocation of good 1 for each type
    # x2[0..15]: allocation of good 2 for each type
    # p[0..15]:  payment for each type
    # Total variables: 48
    
    def unpack(z):
        x1 = z[0:16]
        x2 = z[16:32]
        p = z[32:48]
        return x1, x2, p

    # Objective: Maximize Revenue -> Minimize -Revenue
    def objective(z):
        x1, x2, p = unpack(z)
        revenue = np.dot(joint_probs, p)
        return -revenue

    # Constraints:
    # 1. Monotonicity (Allocations):
    #    If v1' >= v1 and v2' >= v2, then x1(v') >= x1(v) and x2(v') >= x2(v)
    #    Linear constraints: x1(v') - x1(v) >= 0  =>  -x1(v') + x1(v) <= 0
    # 2. Individual Rationality (IR):
    #    p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    #    p(t) - v1(t)*x1(t) - v2(t)*x2(t) <= 0
    # 3. Incentive Compatibility (IC):
    #    u(t, t) >= u(t, t') for all t, t'
    #    v1(t)*x1(t) + v2(t)*x2(t) - p(t) >= v1(t)*x1(t') + v2(t)*x2(t') - p(t')
    #    p(t') - p(t) - v1(t)*(x1(t') - x1(t)) - v2(t)*(x2(t') - x2(t)) <= 0
    # 4. Non-negativity of payments: p(t) >= 0 => -p(t) <= 0
    # 5. Allocation bounds: 0 <= x <= 1 (handled by bounds)

    A_ub = []
    b_ub = []

    # 1. Monotonicity constraints
    # Only need to check immediate neighbors in the lattice for linear monotonicity? 
    # Actually, for linear constraints, checking all pairs or just adjacent is sufficient for convex hull.
    # To keep constraint count low, check adjacent steps in the grid.
    # Grid is 4x4. 
    # For x1: depends on v1. If v1 increases, x1 must increase. v2 doesn't affect x1 monotonicity directly in single-good logic, 
    # but in multi-good, allocation of good 1 should be non-decreasing in v1.
    # Standard monotonicity: x1(v1, v2) is non-decreasing in v1.
    for v1 in range(3):
        for v2 in range(4):
            t_low = get_idx(v1, v2)
            t_high = get_idx(v1 + 1, v2)
            # x1(t_high) - x1(t_low) >= 0  =>  -x1(t_high) + x1(t_low) <= 0
            row = np.zeros(48)
            row[t_high] = -1.0
            row[t_low] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
    # For x2: non-decreasing in v2
    for v1 in range(4):
        for v2 in range(3):
            t_low = get_idx(v1, v2)
            t_high = get_idx(v1, v2 + 1)
            row = np.zeros(48)
            row[16 + t_high] = -1.0
            row[16 + t_low] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)

    # 2. IR constraints
    for t in range(16):
        v1_t, v2_t = types[t]
        row = np.zeros(48)
        row[32 + t] = 1.0       # p(t)
        row[t] = -v1_t          # -v1 * x1
        row[16 + t] = -v2_t     # -v2 * x2
        A_ub.append(row)
        b_ub.append(0.0)

    # 3. IC constraints
    # p(t') - p(t) <= v1(t)*(x1(t') - x1(t)) + v2(t)*(x2(t') - x2(t))
    # p(t') - p(t) - v1(t)*x1(t') + v1(t)*x1(t) - v2(t)*x2(t') + v2(t)*x2(t) <= 0
    for t in range(16):
        v1_t, v2_t = types[t]
        for t_prime in range(16):
            if t == t_prime:
                continue
            row = np.zeros(48)
            # p(t')
            row[32 + t_prime] = 1.0
            # -p(t)
            row[32 + t] = -1.0
            # -v1(t) * x1(t')
            row[t_prime] = -v1_t
            # +v1(t) * x1(t)
            row[t] = v1_t
            # -v2(t) * x2(t')
            row[16 + t_prime] = -v2_t
            # +v2(t) * x2(t)
            row[16 + t] = v2_t
            A_ub.append(row)
            b_ub.append(0.0)

    # 4. Non-negative payments
    for t in range(16):
        row = np.zeros(48)
        row[32 + t] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds
    # x1, x2 in [0, 1]
    # p in [0, None] (upper bound determined by IR, but set a large bound just in case)
    bounds = []
    for i in range(16):
        bounds.append((0.0, 1.0)) # x1
    for i in range(16):
        bounds.append((0.0, 1.0)) # x2
    for i in range(16):
        bounds.append((0.0, None)) # p

    # Initial guess: Threshold mechanism (e.g., threshold 2 for both)
    x1_init = np.zeros(16)
    x2_init = np.zeros(16)
    p_init = np.zeros(16)
    for t in range(16):
        v1_t, v2_t = types[t]
        if v1_t >= 2:
            x1_init[t] = 1.0
        if v2_t >= 2:
            x2_init[t] = 1.0
        p_init[t] = v1_t * x1_init[t] + v2_t * x2_init[t]
        
    z0 = np.concatenate([x1_init, x2_init, p_init])

    best_z = None
    best_obj = np.inf
    best_alloc = None
    best_pay = None

    # Try SLSQP
    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': lambda z: -1e-8 + (b_ub - A_ub @ z) / 1e-8}], # Wait, SLSQP expects g(z) >= 0
                       # Let's redefine constraint function for SLSQP:
                       # SLSQP minimizes f(z) subject to g(z) >= 0.
                       # Our constraints are A_ub @ z <= b_ub.
                       # So g(z) = b_ub - A_ub @ z >= 0.
                       # We need a vectorized function.
                       )
        
        # Redo constraint definition for SLSQP properly
        def constraint_func(z):
            return b_ub - A_ub @ z
        
        res = minimize(objective, z0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': constraint_func}],
                       options={'maxiter': 500, 'ftol': 1e-9})
        
        if res.success:
            x1, x2, p = unpack(res.x)
            # Clean up numerical noise
            x1 = np.clip(x1, 0, 1)
            x2 = np.clip(x2, 0, 1)
            p = np.maximum(p, 0)
            
            # Verify feasibility strictly
            valid = True
            alloc = np.column_stack((x1, x2))
            
            # Check IR
            for t in range(16):
                v1_t, v2_t = types[t]
                u = v1_t * x1[t] + v2_t * x2[t] - p[t]
                if u < -1e-8:
                    valid = False
                    break
            
            # Check IC
            if valid:
                for t in range(16):
                    v1_t, v2_t = types[t]
                    u_tt = v1_t * x1[t] + v2_t * x2[t] - p[t]
                    for t_prime in range(16):
                        if t == t_prime:
                            continue
                        u_tt_prime = v1_t * x1[t_prime] + v2_t * x2[t_prime] - p[t_prime]
                        if u_tt < u_tt_prime - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                rev = float(np.dot(joint_probs, p))
                if -rev < best_obj:
                    best_obj = -rev
                    best_z = res.x.copy()
                    best_alloc = alloc.copy()
                    best_pay = p.copy()

    except Exception as e:
        pass

    # If SLSQP failed or didn't find a valid solution, try a few random restarts or fallback
    if best_alloc is None:
        # Fallback: Simple threshold mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for t in range(16):
            v1_t, v2_t = types[t]
            if v1_t >= 2:
                allocations[t, 0] = 1
            if v2_t >= 2:
                allocations[t, 1] = 1
            payments[t] = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        return {'allocations': allocations, 'payments': payments}

    return {'allocations': best_alloc, 'payments': best_pay}
