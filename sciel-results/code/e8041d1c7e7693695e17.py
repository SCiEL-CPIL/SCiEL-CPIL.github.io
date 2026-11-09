# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using MILP with scipy.optimize.milp.
    """
    # Precompute routes
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    n = 10
    D = len(demands)
    
    # Build edge incidence matrices
    cw_mat = np.zeros((D, n))
    ccw_mat = np.zeros((D, n))
    weights = np.array([w for _, _, w in demands], dtype=np.float64)
    
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_mat[d, e] = 1.0
            e = (e + 1) % n
        for e in range(n):
            if cw_mat[d, e] == 0:
                ccw_mat[d, e] = 1.0
    
    # MILP formulation:
    # Variables: x_d (binary, 1=CCW) for d in 0..D-1, and T (continuous, >= 0)
    # Minimize T
    # Subject to: for each edge e: sum_d w_d * (x_d * ccw_mat[d,e] + (1-x_d) * cw_mat[d,e]) <= T
    #             x_d in {0,1}
    #
    # Expand: w_d * cw_mat[d,e] + w_d * x_d * (ccw_mat[d,e] - cw_mat[d,e]) <= T
    # So: sum_d w_d * cw_mat[d,e] + sum_d w_d * (ccw_mat[d,e] - cw_mat[d,e]) * x_d <= T
    # => sum_d [w_d * (ccw_mat[d,e] - cw_mat[d,e])] * x_d - T <= -sum_d w_d * cw_mat[d,e]
    
    # Objective: minimize T (coefficient 0 for x_d, 1 for T)
    c = np.zeros(D + 1)
    c[D] = 1.0  # T variable
    
    # Constraints: A_ub @ vars <= b_ub
    # For each edge e: sum_d w_d*(ccw_mat[d,e]-cw_mat[d,e])*x_d - T <= -sum_d w_d*cw_mat[d,e]
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    
    for e in range(n):
        for d in range(D):
            A_ub[e, d] = weights[d] * (ccw_mat[d, e] - cw_mat[d, e])
        A_ub[e, D] = -1.0
        b_ub[e] = -np.sum(weights * cw_mat[:, e])
    
    # Bounds: x_d in [0,1], T in [0, inf]
    bounds = [(0, 1)] * D + [(0, None)]
    
    # Integrality: x_d are binary, T is continuous
    integrality = np.ones(D + 1, dtype=np.int8)
    integrality[D] = 0  # T is continuous
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        res = milp(
            c=c,
            constraints=LinearConstraint(A_ub, -np.inf, b_ub),
            integrality=integrality,
            bounds=Bounds(lb=[b[0] for b in bounds], ub=[b[1] if b[1] is not None else 1e10 for b in bounds]),
            options={"time_limit": 500, "mip_rel_gap": 1e-6}
        )
        if res.status == 0 and res.x is not None:
            routing = np.where(res.x[:D] >= 0.5, 1.0, 0.0)
            return routing
    except Exception:
        pass
    
    # Fallback: local search if MILP fails
    def _compute_loads(routing):
        ccw_mask = (routing >= 0.5).astype(np.float64)
        cw_mask = 1.0 - ccw_mask
        loads = np.sum(weights[:, None] * (ccw_mask[:, None] * ccw_mat + cw_mask[:, None] * cw_mat), axis=0)
        return loads
    
    def _local_search_single_flip(routing, deadline, max_rounds=100):
        D_ = len(routing)
        best_routing = routing.copy()
        best_loads = _compute_loads(best_routing)
        best_max_load = np.max(best_loads)
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for d in range(D_):
                if time.time() > deadline:
                    break
                trial_routing = best_routing.copy()
                trial_routing[d] = 1.0 - trial_routing[d]
                trial_loads = _compute_loads(trial_routing)
                trial_max_load = np.max(trial_loads)
                
                if trial_max_load < best_max_load - 1e-12:
                    best_routing = trial_routing
                    best_loads = trial_loads
                    best_max_load = trial_max_load
                    improved = True
            if not improved:
                break
        
        return best_routing, best_max_load
    
    start_time = time.time()
    deadline = start_time + 580
    
    # Try multiple initializations
    rng = np.random.default_rng(42)
    best_routing = None
    best_max_load = float('inf')
    
    inits = [np.zeros(D), np.ones(D)]
    for _ in range(5):
        inits.append(rng.random(D))
    
    for init_sol in inits:
        if time.time() > deadline:
            break
        routing, max_load = _local_search_single_flip(init_sol, deadline)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = routing.copy()
    
    if best_routing is None:
        best_routing = np.zeros(D)
    
    return np.where(best_routing >= 0.5, 1.0, 0.0)


# EVOLVE-BLOCK-END
