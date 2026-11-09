# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using Integer Linear Programming (ILP).
    This is a structurally different approach from exhaustive search, using a
    mathematical optimization solver to find the global optimum.
    """
    # Problem data
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)  # 15
    n_edges = n_nodes # 10 edges

    # Precompute coefficients for the ILP
    # x_i = 1 if demand i is routed CCW, 0 if CW
    # Load on edge e: L_e = C_e + sum_i A_ei * x_i
    # C_e = sum of weights of demands whose CW path includes edge e
    # A_ei = w_i if edge e is in CCW path of i, -w_i if edge e is in CW path of i
    
    C = np.zeros(n_edges)
    A = np.zeros((n_edges, D))
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        cw_edges = set()
        if s <= t:
            for e in range(s, t):
                cw_edges.add(e)
        else:
            for e in range(s, n_nodes):
                cw_edges.add(e)
            for e in range(0, t):
                cw_edges.add(e)
        
        # Determine CCW edges (complement)
        ccw_edges = set(range(n_nodes)) - cw_edges
        
        for e in range(n_edges):
            if e in cw_edges:
                C[e] += w
                A[e, i] = -w
            else:
                # e in ccw_edges
                A[e, i] = w
                # C[e] does not change because base load assumes CW, 
                # but wait.
                # My previous logic:
                # base_load = sum w_i * cw_mask_i
                # load = base_load + sum w_i * (ccw_mask_i - cw_mask_i) * x_i
                # load_e = C_e + sum_i w_i * (ccw_mask_ie - cw_mask_ie) * x_i
                # If e in CW: cw=1, ccw=0. delta = -1. A_ei = -w. C_e includes w.
                # If e in CCW: cw=0, ccw=1. delta = +1. A_ei = +w. C_e does NOT include w.
                
                # So the logic above holds.
                pass

    # Variables: x_1 ... x_D, Z
    # n_vars = D + 1
    n_vars = D + 1
    
    # Objective: Minimize Z
    c = np.zeros(n_vars)
    c[-1] = 1.0
    
    # Constraints:
    # For each edge e:
    # sum_i A_ei * x_i - Z <= -C_e
    # Vector row: [A_e1, ..., A_eD, -1]
    # ub: -C_e
    # lb: -inf
    
    A_mat = np.zeros((n_edges, n_vars))
    for e in range(n_edges):
        A_mat[e, :D] = A[e, :]
        A_mat[e, D] = -1.0
        
    lb = -np.inf * np.ones(n_edges)
    ub = -C
    
    constraints = LinearConstraint(A_mat, lb, ub)
    
    # Bounds
    # x_i in [0, 1], Z in [0, inf)
    integrality = np.ones(n_vars)
    integrality[-1] = 0.0 # Z is continuous (though it will be integer)
    
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    bounds.ub[-1] = np.inf
    
    # Solve
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0: # Optimal
        x_opt = res.x[:D]
        # Convert to routing vector
        # x_i = 1 -> CCW (value >= 0.5)
        # x_i = 0 -> CW (value < 0.5)
        routing = np.where(x_opt > 0.5, 1.0, 0.0)
        return routing
    else:
        # Fallback to all clockwise if solver fails (should not happen)
        return np.zeros(D)

# EVOLVE-BLOCK-END
