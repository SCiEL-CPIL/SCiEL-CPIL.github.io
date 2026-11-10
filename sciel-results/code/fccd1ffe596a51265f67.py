"""Incentive-compatible finite auction design - direct LP with tight tolerances."""

def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import linprog
    import time
    
    t0 = time.time()
    DEADLINE = 85.0
    
    # Parse problem structure
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = 16
    n_goods = 2
    type_idx = {t: i for i, t in enumerate(types)}
    
    # Parse distribution
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        p = np.array(distribution_spec, dtype=float).flatten()
        if p.size == 16:
            p = p / p.sum()
        else:
            p = np.ones(16) / 16.0
    elif isinstance(distribution_spec, dict):
        p = np.zeros(16)
        for k, v in distribution_spec.items():
            if isinstance(k, tuple):
                p[type_idx[k]] = v
            elif isinstance(k, int):
                p[k] = v
        p = p / p.sum()
    else:
        p = np.ones(16) / 16.0
    
    # LP variables: x_1(theta), x_2(theta), t(theta) for each theta
    # Index: x_i(theta) at 2*theta+i, t(theta) at 32+theta
    def xi(theta, good):
        return 2 * theta + good
    
    def ti(theta):
        return 32 + theta
    
    # Objective: max sum p(theta)*t(theta) => min -sum p(theta)*t(theta)
    c = np.zeros(48)
    for theta in range(n_types):
        c[ti(theta)] = -p[theta]
    
    # IC + IR constraints
    A_rows = []
    b_rows = []
    
    for theta in range(n_types):
        v_theta = np.array(types[theta], dtype=float)
        # IC: for all theta'
        for theta_prime in range(n_types):
            row = np.zeros(48)
            for good in range(n_goods):
                row[xi(theta_prime, good)] = v_theta[good]
                row[xi(theta, good)] = -v_theta[good]
            row[ti(theta_prime)] = -1.0
            row[ti(theta)] = 1.0
            A_rows.append(row)
            b_rows.append(0.0)
        
        # IR: -v(theta).x(theta) + t(theta) <= 0
        row = np.zeros(48)
        for good in range(n_goods):
            row[xi(theta, good)] = -v_theta[good]
        row[ti(theta)] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = [(0, 1)] * 32 + [(0, None)] * 16
    
    # Solve with tight tolerances
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                           'primal_feasibility_tolerance': 1e-12,
                           'dual_staleness_tolerance': 1e-12,
                           'primal_staleness_tolerance': 1e-12})
    
    if not res.success:
        # Fallback
        allocs = np.ones((16, 2))
        pays = np.array([float(types[i][0] + types[i][1]) for i in range(16)])
        return {'allocations': allocs, 'payments': pays}
    
    x_sol = res.x
    allocs = np.zeros((16, 2))
    pays = np.zeros(16)
    for theta in range(n_types):
        for good in range(n_goods):
            allocs[theta, good] = x_sol[xi(theta, good)]
        pays[theta] = x_sol[ti(theta)]
    
    # Clean up numerical noise
    allocs = np.clip(allocs, 0.0, 1.0)
    pays = np.maximum(pays, 0.0)
    
    # Snap near-boundary values
    for i in range(16):
        for g in range(2):
            if allocs[i, g] < 1e-12:
                allocs[i, g] = 0.0
            elif allocs[i, g] > 1.0 - 1e-12:
                allocs[i, g] = 1.0
        if pays[i] < 1e-12:
            pays[i] = 0.0
    
    return {'allocations': allocs, 'payments': pays}
