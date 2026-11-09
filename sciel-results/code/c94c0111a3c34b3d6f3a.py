import numpy as np
import time
from scipy import optimize

def ring_routing() -> np.ndarray:
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute edge usage for CW and CCW
    # A[i, e] is the weight added to edge e if demand i is routed CW.
    # B[i, e] is the weight added to edge e if demand i is routed CCW.
    
    A = np.zeros((n_demands, n_nodes))
    B = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # CW edges: s, s+1, ..., t-1
        current = s
        while current != t:
            A[i, current] = w
            current = (current + 1) % n_nodes
        
        # CCW edges: all others
        for e in range(n_nodes):
            if A[i, e] == 0:
                B[i, e] = w
    
    # Variables:
    # x[0..n_demands-1] : binary, 1 if CCW, 0 if CW
    # x[n_demands] : L (max load), continuous >= 0
    
    # Objective: Minimize L
    # c = [0, 0, ..., 0, 1]
    c = np.zeros(n_demands + 1)
    c[n_demands] = 1.0
    
    # Constraints:
    # For each edge e:
    # Sum( A[i, e] * (1 - x_i) + B[i, e] * x_i ) <= L
    # Sum( A[i, e] ) - Sum( A[i, e] * x_i ) + Sum( B[i, e] * x_i ) <= L
    # Sum( A[i, e] ) + Sum( (B[i, e] - A[i, e]) * x_i ) - L <= 0
    
    # Inequality constraints: A_ub @ vars <= b_ub
    # vars = [x_0, ..., x_{n-1}, L]
    
    A_ub = np.zeros((n_nodes, n_demands + 1))
    b_ub = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        # Coefficients for x_i
        for i in range(n_demands):
            A_ub[e, i] = B[i, e] - A[i, e]
        
        # Coefficient for L
        A_ub[e, n_demands] = -1.0
        
        # RHS
        b_ub[e] = np.sum(A[:, e])
        
    # Bounds:
    # x_i in [0, 1]
    # L in [0, inf]
    bounds = []
    for i in range(n_demands):
        bounds.append((0, 1))
    bounds.append((0, None))
    
    # Integrality:
    # x_i are integer (binary)
    # L is continuous
    integrality = np.zeros(n_demands + 1)
    integrality[:n_demands] = 1
    
    try:
        # Use milp (Mixed Integer Linear Programming)
        # This solves the exact problem
        res = optimize.milp(
            c=c,
            constraints=optimize.LinearConstraint(A_ub, ub=b_ub),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0: # Optimal
            x_sol = res.x[:n_demands]
            # Convert to 0.0 or 1.0
            result = np.where(x_sol >= 0.5, 1.0, 0.0)
            return result
        else:
            # Fallback to brute force if milp fails
            pass
            
    except Exception:
        # Fallback if milp not available or errors
        pass

    # Fallback: Brute Force (Current Logic)
    load_contrib = np.zeros((n_demands, 2, n_nodes))
    for i, (s, t, w) in enumerate(demands):
        cw_edges = []
        current = s
        while current != t:
            cw_edges.append(current)
            current = (current + 1) % n_nodes
        
        load_contrib[i, 0, cw_edges] = w
        
        all_edges = np.arange(n_nodes)
        ccw_edges = np.array([e for e in all_edges if e not in cw_edges])
        load_contrib[i, 1, ccw_edges] = w

    base_load = np.sum(load_contrib[:, 0, :], axis=0)
    delta = load_contrib[:, 1, :] - load_contrib[:, 0, :]
    
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    for mask in range(1 << n_demands):
        if time.time() > deadline:
            break
            
        loads = base_load.copy()
        m = mask
        i = 0
        while m:
            if m & 1:
                loads += delta[i]
            m >>= 1
            i += 1
        
        current_max = np.max(loads)
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask
            
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
            
    return result
