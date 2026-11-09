"""Rewritten program: direct LP solve for incentive-compatible finite auction design."""
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Args:
        distribution_spec: spec describing the distribution (one of three fixed types)
        support: support of values (each in {0,1,2,3})
        constraints: constraint dict
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse the problem structure
    # One additive bidder, two goods, each value in {0,1,2,3}
    # So there are 4*4 = 16 possible value profiles (v1, v2)
    # Each report is also (r1, r2) with ri in {0,1,2,3}, so 16 reports
    
    # The mechanism maps each report (r1, r2) to:
    #   allocation x(r1, r2) in [0,1]^2 (probability of allocating each good)
    #   payment p(r1, r2) >= 0
    
    # We need to solve an LP to maximize expected revenue subject to:
    # 1. Feasibility: x in [0,1]^2, p >= 0
    # 2. Individual rationality: expected utility >= 0 for each true type
    # 3. Incentive compatibility: no report gives higher utility than truthful
    # 4. Regret <= 1e-8 (numerical tolerance)
    
    # Determine value support
    values = [0, 1, 2, 3]
    n_vals = len(values)
    n_reports = n_vals * n_vals  # 16
    
    # Enumerate all reports
    reports = [(v1, v2) for v1 in values for v2 in values]
    report_idx = {(v1, v2): i for i, (v1, v2) in enumerate(reports)}
    
    # Parse distribution
    # distribution_spec could be various formats; handle common cases
    if isinstance(distribution_spec, dict):
        # Try to extract probabilities
        if 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, dict):
                probs = np.zeros(n_reports)
                for key, prob in dist.items():
                    if isinstance(key, (list, tuple)):
                        idx = report_idx.get(tuple(key), None)
                        if idx is not None:
                            probs[idx] = prob
                    else:
                        # Try to parse as index
                        idx = int(key)
                        if 0 <= idx < n_reports:
                            probs[idx] = prob
            else:
                probs = np.array(dist, dtype=float)
        else:
            # Default: uniform
            probs = np.ones(n_reports) / n_reports
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_reports:
            # Maybe it's per-gain probabilities
            if len(probs) == n_vals:
                probs = np.outer(probs, probs).flatten()
            else:
                probs = np.ones(n_reports) / n_reports
    else:
        probs = np.ones(n_reports) / n_reports
    
    # Normalize probabilities
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_reports) / n_reports
    
    # Ensure probs has correct length
    if len(probs) != n_reports:
        probs = np.ones(n_reports) / n_reports
    
    # Variables:
    # x[i, g] for i in 0..15, g in 0..1  -> 32 variables
    # p[i] for i in 0..15                -> 16 variables
    # Total: 48 variables
    # Order: [x[0,0], x[0,1], x[1,0], x[1,1], ..., x[15,0], x[15,1], p[0], p[1], ..., p[15]]
    
    n_x = n_reports * 2  # 32
    n_p = n_reports      # 16
    n_vars = n_x + n_p   # 48
    
    def x_var(i, g):
        return i * 2 + g
    
    def p_var(i):
        return n_x + i
    
    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    # In linprog form: minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[p_var(i)] = -probs[i]
    
    # Constraints:
    # 1. x[i, g] <= 1 for all i, g (upper bound handled by bounds)
    # 2. x[i, g] >= 0 (lower bound handled by bounds)
    # 3. p[i] >= 0 (lower bound handled by bounds)
    # 4. Individual rationality: for each true type t, E_{r~report|truthful}[v(t,r)*x(r) - p(r)] >= 0
    #    Actually IR means: for each true type t, the expected utility when reporting truthfully >= 0
    #    u(t, t) = sum_g t_g * x[t, g] - p[t] >= 0
    #    i.e., p[t] <= sum_g t_g * x[t, g]
    # 5. Incentive compatibility: for each true type t and each report r,
    #    u(t, r) <= u(t, t)
    #    sum_g t_g * x[r, g] - p[r] <= sum_g t_g * x[t, g] - p[t]
    #    sum_g t_g * x[r, g] - p[r] - sum_g t_g * x[t, g] + p[t] <= 0
    
    # Let's build the constraint matrix A_ub and b_ub using sparse representation
    # For IR: p[t] - sum_g t_g * x[t, g] <= 0
    # For IC: sum_g t_g * x[r, g] - p[r] - sum_g t_g * x[t, g] + p[t] <= 0
    
    n_constraints = n_reports + n_reports * n_reports  # IR + IC
    A_sp = lil_matrix((n_constraints, n_vars), dtype=float)
    b_ub = np.zeros(n_constraints)
    
    # IR constraints: one per true type t
    for t in range(n_reports):
        tv1, tv2 = reports[t]
        row_idx = t
        A_sp[row_idx, p_var(t)] = 1.0
        A_sp[row_idx, x_var(t, 0)] = -tv1
        A_sp[row_idx, x_var(t, 1)] = -tv2
    
    # IC constraints: for each true type t and each report r
    constraint_offset = n_reports
    for t in range(n_reports):
        tv1, tv2 = reports[t]
        for r in range(n_reports):
            row_idx = constraint_offset + t * n_reports + r
            A_sp[row_idx, x_var(r, 0)] = tv1
            A_sp[row_idx, x_var(r, 1)] = tv2
            A_sp[row_idx, p_var(r)] = -1.0
            A_sp[row_idx, x_var(t, 0)] = -tv1
            A_sp[row_idx, x_var(t, 1)] = -tv2
            A_sp[row_idx, p_var(t)] = 1.0
    
    A_ub = A_sp.tocsr()
    
    # Bounds:
    # x[i, g] in [0, 1]
    # p[i] in [0, inf)
    bounds = []
    for i in range(n_reports):
        for g in range(2):
            bounds.append((0.0, 1.0))
    for i in range(n_reports):
        bounds.append((0.0, None))
    
    # Solve LP with tight tolerances
    best_allocations = np.zeros((n_reports, 2))
    best_payments = np.zeros(n_reports)
    best_revenue = -1.0
    
    try:
        # First solve with default tolerances
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                        options={'presolve': True, 'primal_feasibility_tolerance': 1e-10,
                                 'dual_feasibility_tolerance': 1e-10,
                                 'time_limit': 40.0})
        
        if result.success:
            x_sol = result.x[:n_x].reshape(n_reports, 2)
            p_sol = result.x[n_x:]
            x_sol = np.clip(x_sol, 0.0, 1.0)
            p_sol = np.clip(p_sol, 0.0, None)
            
            # Verify and repair if needed
            x_sol, p_sol = _verify_and_repair(x_sol, p_sol, reports, n_reports)
            
            revenue = np.dot(probs, p_sol)
            if revenue > best_revenue:
                best_revenue = revenue
                best_allocations = x_sol.copy()
                best_payments = p_sol.copy()
        
        # Check time - if we have time, try a second pass with tighter tolerances
        elapsed = time.time() - t0
        if elapsed < 60.0:
            try:
                result2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                options={'presolve': True, 'primal_feasibility_tolerance': 1e-12,
                                         'dual_feasibility_tolerance': 1e-12,
                                         'time_limit': 30.0})
                if result2.success:
                    x_sol2 = result2.x[:n_x].reshape(n_reports, 2)
                    p_sol2 = result2.x[n_x:]
                    x_sol2 = np.clip(x_sol2, 0.0, 1.0)
                    p_sol2 = np.clip(p_sol2, 0.0, None)
                    x_sol2, p_sol2 = _verify_and_repair(x_sol2, p_sol2, reports, n_reports)
                    revenue2 = np.dot(probs, p_sol2)
                    if revenue2 > best_revenue:
                        best_revenue = revenue2
                        best_allocations = x_sol2.copy()
                        best_payments = p_sol2.copy()
            except Exception:
                pass
        
    except Exception:
        pass
    
    return {'allocations': best_allocations, 'payments': best_payments}


def _verify_and_repair(x_sol, p_sol, reports, n_reports):
    """Verify IC and IR constraints, repair if violations exist."""
    x_sol = np.clip(x_sol, 0.0, 1.0)
    p_sol = np.clip(p_sol, 0.0, None)
    
    # Compute utilities for all (t, r) pairs
    # u(t, r) = t[0]*x[r,0] + t[1]*x[r,1] - p[r]
    max_violation = 0.0
    
    for t in range(n_reports):
        tv1, tv2 = reports[t]
        u_tt = tv1 * x_sol[t, 0] + tv2 * x_sol[t, 1] - p_sol[t]
        
        # IR check
        if u_tt < 0:
            max_violation = max(max_violation, -u_tt)
        
        # IC check
        for r in range(n_reports):
            u_tr = tv1 * x_sol[r, 0] + tv2 * x_sol[r, 1] - p_sol[r]
            violation = u_tr - u_tt
            if violation > max_violation:
                max_violation = violation
    
    if max_violation <= 1e-8:
        return x_sol, p_sol
    
    # Repair: iteratively reduce payments to fix violations
    # Use a more careful approach: find the minimal payment reduction needed
    for iteration in range(50):
        max_violation = 0.0
        worst_t, worst_r = -1, -1
        
        for t in range(n_reports):
            tv1, tv2 = reports[t]
            u_tt = tv1 * x_sol[t, 0] + tv2 * x_sol[t, 1] - p_sol[t]
            
            if u_tt < 0:
                if -u_tt > max_violation:
                    max_violation = -u_tt
                    worst_t, worst_r = t, t
            
            for r in range(n_reports):
                u_tr = tv1 * x_sol[r, 0] + tv2 * x_sol[r, 1] - p_sol[r]
                violation = u_tr - u_tt
                if violation > max_violation:
                    max_violation = violation
                    worst_t, worst_r = t, r
        
        if max_violation <= 1e-8:
            break
        
        # Reduce payment at worst_r by the violation amount
        p_sol[worst_r] = max(0.0, p_sol[worst_r] - max_violation)
    
    # Final clamping
    x_sol = np.clip(x_sol, 0.0, 1.0)
    p_sol = np.clip(p_sol, 0.0, None)
    
    return x_sol, p_sol
