import numpy as np
import time
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 500  # 500s safety margin
    
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge masks for CW and CCW routing for each demand
    # CW from s to t: edges s, s+1, ..., t-1 (mod n)
    # CCW from s to t: all other edges
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw = np.zeros(n_nodes)
        if s == t:
            # Edge case: all edges (shouldn't happen in this problem)
            cw[:] = 1.0
        else:
            edge = s
            while edge != t:
                cw[edge] = 1.0
                edge = (edge + 1) % n_nodes
        cw_masks.append(cw)
        ccw = 1.0 - cw
        ccw_masks.append(ccw)
    
    cw_masks = np.array(cw_masks)  # shape (D, n_nodes)
    ccw_masks = np.array(ccw_masks)  # shape (D, n_nodes)
    weights = np.array([d[2] for d in demands], dtype=np.float64)  # shape (D,)
    
    # Precompute for efficient evaluation
    delta_masks = ccw_masks - cw_masks  # shape (D, n_nodes)
    weighted_cw = cw_masks * weights[:, None]  # shape (D, n_nodes)
    weighted_delta = delta_masks * weights[:, None]  # shape (D, n_nodes)
    base_load = weighted_cw.sum(axis=0)  # shape (n_nodes,)
    
    def evaluate(routing):
        """Compute max edge load given routing vector (0=CW, 1=CCW)."""
        # total_load = base_load + sum_d weighted_delta[d] * routing[d]
        delta_load = (weighted_delta * routing[:, None]).sum(axis=0)
        total_load = base_load + delta_load
        return np.max(total_load)
    
    # Strategy 1: Try LP relaxation (continuous) to get a good starting point
    # Variables: x_0 ... x_{D-1} (routing decisions, continuous [0,1]), t (max load)
    # Minimize t
    # Subject to: for each edge e: sum_d w_d * mask_d,e * x_d + ... <= t
    #             0 <= x_d <= 1
    #             t >= 0
    
    # Load on edge e = sum_d w_d * (cw_mask_d,e * (1-x_d) + ccw_mask_d,e * x_d)
    #                = sum_d w_d * cw_mask_d,e + sum_d w_d * (ccw_mask_d,e - cw_mask_d,e) * x_d
    #                = base_load[e] + sum_d weighted_delta[d,e] * x_d
    # Constraint: base_load[e] + sum_d weighted_delta[d,e] * x_d <= t
    #            => -sum_d weighted_delta[d,e] * x_d + t >= base_load[e]
    #            => sum_d (-weighted_delta[d,e]) * x_d - t <= -base_load[e]
    
    # Number of variables: D + 1 (x_0..x_{D-1}, t)
    n_vars = D + 1
    
    # Objective: minimize t => c = [0, 0, ..., 0, 1]
    c = np.zeros(n_vars)
    c[-1] = 1.0
    
    # Constraints: for each edge e (n_nodes constraints)
    # sum_d (-weighted_delta[d,e]) * x_d - t <= -base_load[e]
    A_ub = np.zeros((n_nodes, n_vars))
    b_ub = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        for d in range(D):
            A_ub[e, d] = -weighted_delta[d, e]
        A_ub[e, -1] = -1.0  # coefficient of t
        b_ub[e] = -base_load[e]
    
    # Bounds: 0 <= x_d <= 1 for d in range(D), t >= 0
    bounds = [(0, 1)] * D + [(0, None)]
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            lp_routing = np.array(result.x[:D], dtype=np.float64)
            lp_routing = (lp_routing >= 0.5).astype(np.float64)
            lp_score = evaluate(lp_routing)
        else:
            lp_routing = None
            lp_score = float('inf')
    except Exception:
        lp_routing = None
        lp_score = float('inf')
    
    # Strategy 2: Brute force all 2^15 combinations (32768 is very fast)
    best_max_load = float('inf')
    best_routing = np.zeros(D)
    
    # Start with a good initial guess if LP worked
    if lp_routing is not None:
        best_routing = lp_routing.copy()
        best_max_load = lp_score
    
    # Also try all-CW and all-CCW
    all_cw = np.zeros(D)
    all_ccw = np.ones(D)
    for init in [all_cw, all_ccw]:
        s = evaluate(init)
        if s < best_max_load:
            best_max_load = s
            best_routing = init.copy()
    
    # Brute force
    for mask in range(1 << D):
        if time.time() > deadline:
            break
        
        loads = base_load.copy()
        
        for i in range(D):
            if (mask >> i) & 1:
                # Counter-clockwise: add weighted_delta
                loads += weighted_delta[i]
            else:
                # Clockwise: no additional load (already in base_load)
                pass
        
        current_max = np.max(loads)
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = np.zeros(D)
            for i in range(D):
                if (mask >> i) & 1:
                    best_routing[i] = 1.0
    
    return best_routing
