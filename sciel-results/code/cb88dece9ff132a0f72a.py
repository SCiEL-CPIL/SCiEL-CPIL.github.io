import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem.
    Attempts to use scipy.optimize.milp for an exact MIP solution.
    Falls back to exhaustive search if milp is unavailable or fails.
    """
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2)
    ]

    n_nodes = 10
    n_demands = len(demands)

    # Pre-compute load vectors for clockwise and counter-clockwise routing
    cw_loads = np.zeros((n_demands, n_nodes))
    ccw_loads = np.zeros((n_demands, n_nodes))

    for i, (s, t, w) in enumerate(demands):
        # Clockwise: edges s, s+1, ..., t-1 (mod n_nodes)
        current = s
        while current != t:
            cw_loads[i, current] += w
            current = (current + 1) % n_nodes

        # Counter-clockwise: all other edges
        for e in range(n_nodes):
            if cw_loads[i, e] == 0:
                ccw_loads[i, e] = w

    # delta_loads[i, e] = ccw_loads[i, e] - cw_loads[i, e]
    # base_load[e] = sum_i cw_loads[i, e]
    # Load for mask m = base_load + sum_{i in bits(m)} delta_loads[i]
    
    delta_loads = ccw_loads - cw_loads
    base_load = cw_loads.sum(axis=0)

    # Try MIP solver first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Variables: x_0 ... x_14 (binary), L (continuous)
        # x_i = 1 means CCW, 0 means CW
        
        # Load on edge e = base_load[e] + sum_i x_i * delta_loads[i,e]
        # Constraint: base_load[e] + sum_i x_i * delta_loads[i,e] <= L
        # => sum_i x_i * delta_loads[i,e] - L <= -base_load[e]
        
        # Objective: Minimize L
        c = np.zeros(n_demands + 1)
        c[-1] = 1.0
        
        A = np.zeros((n_nodes, n_demands + 1))
        b = np.zeros(n_nodes)
        
        for e in range(n_nodes):
            A[e, :n_demands] = delta_loads[:, e]
            A[e, -1] = -1.0
            b[e] = -base_load[e]
            
        constraints = LinearConstraint(A, -np.inf, b)
        
        # Bounds: x_i in [0,1] and integer, L >= 0
        integrality = np.ones(n_demands + 1)
        integrality[-1] = 0 # L is continuous
        
        bounds = Bounds(lb=np.zeros(n_demands + 1), ub=np.ones(n_demands + 1))
        bounds.ub[-1] = np.inf # L can be large
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            x = res.x[:n_demands]
            # Round to 0 or 1 to be safe
            routing = (x > 0.5).astype(np.float64)
            return routing
        else:
            # Fallback to exhaustive search
            pass
            
    except Exception:
        # Fallback to exhaustive search
        pass

    # Exhaustive search fallback
    total_masks = 1 << n_demands
    masks = np.arange(total_masks, dtype=np.uint16)
    bits = ((masks[:, None] >> np.arange(n_demands, dtype=np.uint16)) & 1).astype(np.float64)
    loads = base_load[None, :] + bits @ delta_loads
    max_loads = loads.max(axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    routing = ((best_mask >> np.arange(n_demands, dtype=np.uint16)) & 1).astype(np.float64)
    
    return routing
