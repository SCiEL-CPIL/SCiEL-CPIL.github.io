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
    weights = [demands[i][2] for i in range(D)]

    # Helper to compute max load for a given routing vector
    def maxload_of(r):
        loads = np.zeros(n)
        for i in range(D):
            edges = ccw_edges[i] if r[i] >= 0.5 else cw_edges[i]
            for e in edges:
                loads[e] += weights[i]
        return float(np.max(loads))

    best_load = float('inf')
    best_routing = None

    # ---- Phase 0: LP Relaxation + Rounding ----
    # This often finds a very good starting point, potentially optimal or near-optimal.
    if time.time() < deadline - safety:
        try:
            lp_routing = _solve_lp(n, D, cw_edges, ccw_edges, weights)
            if lp_routing is not None:
                lp_load = maxload_of(lp_routing)
                if lp_load < best_load:
                    best_load = lp_load
                    best_routing = lp_routing
        except Exception:
            pass

    # ---- Phase 1: exact global enumeration over all 2^D routings ----
    # Represent routing as a bitmask: bit i = 1 -> counter-clockwise.
    # If LP didn't find a perfect solution or we want to ensure global optimality,
    # we run enumeration. Since D=15, 2^15 is small enough.
    # We can prune enumeration if we find a solution with load <= LP lower bound,
    # but LP lower bound is just res.fun.
    # Actually, since 2^15 is small, we can just run it to guarantee the best integer solution.
    # However, the LP solution might be the same. Let's run enumeration to be safe and precise.
    
    # Optimization: If LP solution is integer, it's optimal.
    # Let's check if the LP solution was integer.
    # But we lost the fractional values in _solve_lp return. 
    # Let's just run the Gray code enumeration. It's fast.
    
    loads = np.zeros(n)
    prev_gray = 0
    # start at mask 0 (all clockwise)
    for i in range(D):
        for e in cw_edges[i]:
            loads[e] += weights[i]
    cur_max = float(np.max(loads))
    if cur_max < best_load:
        best_load = cur_max
        best_routing = np.zeros(D)

    total = 1 << D
    for step in range(1, total):
        gray = step ^ (step >> 1)
        # find changed bit
        diff = gray ^ prev_gray
        bit = (diff & -diff).bit_length() - 1
        # determine new direction for bit
        is_ccw = (gray >> bit) & 1
        # remove old contribution, add new
        if is_ccw:
            # was clockwise, now ccw
            for e in cw_edges[bit]:
                loads[e] -= weights[bit]
            for e in ccw_edges[bit]:
                loads[e] += weights[bit]
        else:
            # was ccw, now clockwise
            for e in ccw_edges[bit]:
                loads[e] -= weights[bit]
            for e in cw_edges[bit]:
                loads[e] += weights[bit]
        prev_gray = gray
        cur_max = float(np.max(loads))
        if cur_max < best_load:
            best_load = cur_max
            # Construct routing
            r = np.zeros(D)
            for i in range(D):
                r[i] = 1.0 if (gray >> i) & 1 else 0.0
            best_routing = r
        
        if time.time() > deadline - safety:
            break

    # ---- Phase 2: local-search polish (single + double flips) from best ----
    # Even though enumeration is exhaustive, local search can help if enumeration was cut short.
    # If enumeration completed, this is redundant but safe.
    if best_routing is None:
        # Fallback if enumeration didn't run or failed
        best_routing = np.zeros(D)
        best_load = maxload_of(best_routing)

    routing = best_routing.copy()
    cur_load = best_load

    improved = True
    while improved and (time.time() < deadline - safety):
        improved = False
        # single flips
        for i in range(D):
            trial = routing.copy()
            trial[i] = 0.0 if trial[i] >= 0.5 else 1.0
            v = maxload_of(trial)
            if v < cur_load - 1e-12:
                routing = trial
                cur_load = v
                improved = True
        # double flips
        for i in range(D):
            for j in range(i + 1, D):
                trial = routing.copy()
                trial[i] = 0.0 if trial[i] >= 0.5 else 1.0
                trial[j] = 0.0 if trial[j] >= 0.5 else 1.0
                v = maxload_of(trial)
                if v < cur_load - 1e-12:
                    routing = trial
                    cur_load = v
                    improved = True

    # Final exact verification
    final = maxload_of(routing)
    if final < best_load:
        best_load = final
        best_routing = routing

    return best_routing


# EVOLVE-BLOCK-END
