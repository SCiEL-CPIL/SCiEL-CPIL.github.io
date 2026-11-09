# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def _build_data():
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    cw = []
    ccw = []
    for (s, t, w) in demands:
        if s == t:
            cw_edges = []
        else:
            cw_edges = []
            e = s
            while e != t:
                cw_edges.append(e)
                e = (e + 1) % n
        ccw_edges = [e for e in range(n) if e not in set(cw_edges)]
        cw.append((cw_edges, w))
        ccw.append((ccw_edges, w))
    return demands, cw, ccw, n, D


def _compute_max_load(routing, cw, ccw, n):
    loads = np.zeros(n)
    for i in range(len(routing)):
        if routing[i] >= 0.5:
            edges, w = ccw[i]
        else:
            edges, w = cw[i]
        for e in edges:
            loads[e] += w
    return float(np.max(loads)) if n > 0 else 0.0


def _solve_lp(n, D, cw_edges, ccw_edges, weights):
    """
    Solve the LP relaxation:
    Minimize t
    subject to:
      For each edge e: sum over demands i using e of (w_i * x_i if ccw else w_i * (1-x_i)) <= t
      0 <= x_i <= 1
    
    We reformulate the edge constraints.
    Let x_i be the fraction routed CCW.
    Load on edge e = sum_{i: e in cw_i} w_i * (1-x_i) + sum_{i: e in ccw_i} w_i * x_i
    = sum_{i: e in cw_i} w_i - sum_{i: e in cw_i} w_i * x_i + sum_{i: e in ccw_i} w_i * x_i
    = C_e + sum_i (A_{e,i} * x_i)
    where C_e = sum_{i: e in cw_i} w_i
    and A_{e,i} = w_i if e in ccw_i else -w_i if e in cw_i else 0
    
    Constraint: C_e + sum_i A_{e,i} x_i - t <= 0
    Variables: x_0...x_{D-1}, t
    Objective: t (minimize) -> c = [0, ..., 0, 1]
    Bounds: 0 <= x_i <= 1, t >= 0
    """
    # Variables: [x_0, ..., x_{D-1}, t]
    c = np.zeros(D + 1)
    c[-1] = 1.0  # Minimize t
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For each edge e: sum_i A_{e,i} x_i - t <= -C_e
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    
    for e in range(n):
        C_e = 0
        for i in range(D):
            if e in cw_edges[i]:
                A_ub[e, i] = -weights[i]
                C_e += weights[i]
            if e in ccw_edges[i]:
                A_ub[e, i] += weights[i]
        A_ub[e, -1] = -1.0
        b_ub[e] = -C_e
    
    bounds = [(0, 1)] * D + [(0, None)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status == 0:
        x_frac = res.x[:D]
        # Rounding strategy:
        # 1. Simple threshold 0.5
        # 2. Check if any x is very close to 0 or 1, fix them, re-solve? 
        #    For now, simple thresholding and then local search polish is robust.
        routing = np.zeros(D)
        for i in range(D):
            if x_frac[i] >= 0.5:
                routing[i] = 1.0
            else:
                routing[i] = 0.0
        return routing
    else:
        return None


def ring_routing() -> np.ndarray:
    start = time.time()
    deadline = start + 600.0
    safety = 30.0  # return at least 30s before hard limit

    demands, cw, ccw, n, D = _build_data()

    # Precompute per-demand edge lists as lists of ints for fast access.
    cw_edges = [cw[i][0] for i in range(D)]
    ccw_edges = [ccw[i][0] for i in range(D)]
    weights = np.array([demands[i][2] for i in range(D)], dtype=np.float64)

    # Helper to compute max load for a given routing vector
    def maxload_of(r):
        loads = np.zeros(n)
        for i in range(D):
            edges = ccw_edges[i] if r[i] >= 0.5 else cw_edges[i]
            for e in edges:
                loads[e] += weights[i]
        return float(np.max(loads))

    # Exhaustive enumeration using Gray code for O(1) update per step
    loads = np.zeros(n)
    # Start with all clockwise (mask = 0)
    for i in range(D):
        for e in cw_edges[i]:
            loads[e] += weights[i]
    
    best_load = float(np.max(loads))
    best_mask = 0

    total = 1 << D
    prev_gray = 0
    
    for step in range(1, total):
        if time.time() > deadline - safety:
            break
        
        gray = step ^ (step >> 1)
        diff = gray ^ prev_gray
        bit = (diff & -diff).bit_length() - 1
        is_ccw = (gray >> bit) & 1
        
        if is_ccw:
            for e in cw_edges[bit]:
                loads[e] -= weights[bit]
            for e in ccw_edges[bit]:
                loads[e] += weights[bit]
        else:
            for e in ccw_edges[bit]:
                loads[e] -= weights[bit]
            for e in cw_edges[bit]:
                loads[e] += weights[bit]
        
        prev_gray = gray
        cur_max = float(np.max(loads))
        if cur_max < best_load - 1e-15:
            best_load = cur_max
            best_mask = gray

    # Construct final routing from best_mask
    best_routing = np.zeros(D)
    for i in range(D):
        best_routing[i] = 1.0 if (best_mask >> i) & 1 else 0.0

    # Final verification
    final_load = maxload_of(best_routing)
    if final_load < best_load:
        best_load = final_load

    return best_routing


# EVOLVE-BLOCK-END
