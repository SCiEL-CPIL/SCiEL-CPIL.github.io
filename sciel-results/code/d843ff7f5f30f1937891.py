import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def ring_routing() -> np.ndarray:
    N = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge usage
    # cw_mask[i, e] = 1 if demand i uses edge e in CW
    # ccw_mask[i, e] = 1 if demand i uses edge e in CCW
    cw_mask = np.zeros((D, N))
    ccw_mask = np.zeros((D, N))
    weights = np.array([w for (_, _, w) in demands], dtype=float)
    
    for i, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_mask[i, e] = 1.0
            e = (e + 1) % N
        # CCW is the complement
        ccw_mask[i, :] = 1.0 - cw_mask[i, :]
        
    # ILP Formulation
    # Variables: x[0..D-1] (binary), T (continuous)
    # Minimize T
    # Constraints: For each edge e, sum(w_i * (cw_mask[i,e]*(1-x_i) + ccw_mask[i,e]*x_i)) <= T
    
    # Rewrite load expression:
    # load_e = sum_i w_i * cw_mask[i,e] - sum_i w_i * cw_mask[i,e] * x_i + sum_i w_i * ccw_mask[i,e] * x_i
    # load_e = sum_i w_i * cw_mask[i,e] + sum_i w_i * (ccw_mask[i,e] - cw_mask[i,e]) * x_i
    # Let A_e = sum_i w_i * cw_mask[i,e]
    # Let B_ei = w_i * (ccw_mask[i,e] - cw_mask[i,e])
    # Constraint: A_e + sum_i B_ei * x_i <= T
    
    # Objective: min T
    # c = [0, ..., 0, 1] (D zeros, then 1 for T)
    
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    # Constraints matrix A_ub x <= b_ub
    # We have N constraints (one per edge)
    # Each constraint: sum_i B_ei * x_i - T <= -A_e
    # Or: -sum_i B_ei * x_i + T >= A_e
    # Standard form for milp: A_ub @ x <= b_ub
    
    A_ub = np.zeros((N, D + 1))
    b_ub = np.zeros(N)
    
    for e in range(N):
        # Coefficients for x_i
        for i in range(D):
            A_ub[e, i] = -weights[i] * (ccw_mask[i, e] - cw_mask[i, e])
        # Coefficient for T
        A_ub[e, D] = 1.0
        # RHS
        b_ub[e] = weights @ cw_mask[:, e]
        
    # Bounds
    # x_i in [0, 1]
    # T in [0, inf)
    bounds = Bounds(lb=np.concatenate([np.zeros(D), [0]]), 
                    ub=np.concatenate([np.ones(D), [np.inf]]))
    
    # Integrality
    integrality = np.concatenate([np.ones(D), [0]])
    
    try:
        res = milp(c=c, constraints=LinearConstraint(A_ub, -np.inf, b_ub), 
                   integrality=integrality, bounds=bounds)
        if res.success:
            x = res.x[:D]
            # Map to routing array
            # x_i = 1 means CCW, 0 means CW
            routing = x
            return routing
        else:
            # Fallback to exhaustive if MILP fails
            pass
    except Exception:
        pass

    # Fallback: Exhaustive Search
    best_routing = None
    best_max = float('inf')
    
    def compute_loads_fast(routing_binary):
        r = routing_binary.astype(float)
        cw_contrib = (weights * (1.0 - r)) @ cw_mask
        ccw_contrib = (weights * r) @ ccw_mask
        return cw_contrib + ccw_contrib

    total_combos = 1 << D
    for mask in range(total_combos):
        routing = np.array([(mask >> d) & 1 for d in range(D)], dtype=float)
        loads = compute_loads_fast(routing)
        ml = loads.max()
        if ml < best_max:
            best_max = ml
            best_routing = routing.copy()
            
    return best_routing
