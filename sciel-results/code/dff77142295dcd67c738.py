import numpy as np
import scipy.optimize as so
import time

def ring_routing() -> np.ndarray:
    n_nodes = 10
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
        (5, 2, 2),
    ]
    D = len(demands)
    
    # Calculate CW and CCW edge sets for each demand
    cw_edges_list = []
    ccw_edges_list = []
    for s, t, w in demands:
        cw_set = set()
        e = s
        while e != t:
            cw_set.add(e)
            e = (e + 1) % n_nodes
        cw_edges_list.append(cw_set)
        
        ccw_set = set(range(n_nodes)) - cw_set
        ccw_edges_list.append(ccw_set)
        
    # Base loads (all CW)
    base_loads = np.zeros(n_nodes, dtype=np.float64)
    # Coefficients for ILP: A[e, i] = w_i * (1 if e in CCW_i else 0 - 1 if e in CW_i else 0)
    A_coeffs = np.zeros((n_nodes, D), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        for e in range(n_nodes):
            in_cw = 1 if e in cw_edges_list[i] else 0
            in_ccw = 1 if e in ccw_edges_list[i] else 0
            A_coeffs[e, i] = w * (in_ccw - in_cw)
            
            if in_cw:
                base_loads[e] += w

    # ILP Formulation
    # Variables: x[0..14] (binary), t[15] (continuous)
    # Minimize t
    c = np.zeros(D + 1)
    c[-1] = 1.0
    
    # Constraints: Load_e <= t  =>  Sum(A[e, i] * x_i) - t <= -base_loads[e]
    # Matrix shape: (n_nodes, D+1)
    A_ub = np.zeros((n_nodes, D + 1), dtype=np.float64)
    A_ub[:, :D] = A_coeffs
    A_ub[:, D] = -1.0
    
    b_ub = -base_loads
    
    # Integrality: 1 for x, 0 for t
    integrality = np.ones(D + 1)
    integrality[-1] = 0
    
    # Bounds: x in [0, 1], t in [0, inf]
    bounds = [(0, 1)] * D + [(0, None)]
    
    start_time = time.time()
    
    try:
        res = so.milp(c=c, constraints=(so.LinearConstraint(A_ub, ub=b_ub),), 
                      integrality=integrality, bounds=bounds, 
                      options={'time_limit': 590})
        
        if res.status == 0:
            x_sol = res.x[:D]
            # Convert to routing vector
            routing = np.where(x_sol > 0.5, 1.0, 0.0)
            return routing
        else:
            # Fallback to exhaustive search if ILP fails
            pass
    except Exception:
        # Fallback to exhaustive search if ILP fails
        pass

    # Fallback: Exhaustive Search (same as original logic)
    diff_loads = A_coeffs # This is (CCW - CW) * w
    
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)
    
    current_loads = base_loads.copy()
    current_routing = np.zeros(D, dtype=np.float64)
    
    for i in range(1 << D):
        max_load = np.max(current_loads)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = current_routing.copy()
            
        if i < (1 << D) - 1:
            g_i = i ^ (i >> 1)
            g_next = (i + 1) ^ ((i + 1) >> 1)
            changed_bit = g_i ^ g_next
            bit_idx = (changed_bit.bit_length() - 1)
            
            if (g_next >> bit_idx) & 1:
                current_loads += diff_loads[:, bit_idx]
                current_routing[bit_idx] = 1.0
            else:
                current_loads -= diff_loads[:, bit_idx]
                current_routing[bit_idx] = 0.0
                
    return best_routing
