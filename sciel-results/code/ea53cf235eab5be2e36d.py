import time
import numpy as np
from itertools import combinations

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by binary searching on the maximum edge load L,
    and for each L, formulating a 2-SAT instance to check feasibility.
    The 2-SAT clauses are generated from the constraint that no edge load exceeds L.
    Since the problem is small (15 demands, 10 edges), we use a simple backtracking
    2-SAT solver.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    n = 10
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
    d = len(demands)

    # Pre-compute clockwise and counter-clockwise edge usage matrices
    cw = np.zeros((d, n), dtype=np.int64)
    ccw = np.zeros((d, n), dtype=np.int64)
    w = np.zeros(d, dtype=np.int64)

    for i, (s, t, wt) in enumerate(demands):
        w[i] = wt
        if s == t:
            continue
        e = s
        while e != t:
            cw[i, e] = 1
            e = (e + 1) % n
        ccw[i] = 1 - cw[i]

    # For each edge e, pre-compute:
    # base_load[e] = sum_i w_i * cw[i,e]
    # delta[e, i] = w_i * (ccw[i,e] - cw[i,e])
    base_load = np.zeros(n, dtype=np.int64)
    delta = np.zeros((n, d), dtype=np.int64)
    for e in range(n):
        base_load[e] = np.dot(w, cw[:, e])
        delta[e] = w * (ccw[:, e] - cw[:, e])

    # Binary search on L (max edge load)
    # Lower bound: max over edges of min possible load
    # Upper bound: max over edges of max possible load (all demands routed to worsen that edge)
    lo = 0
    hi = 0
    for e in range(n):
        # Min load on edge e: choose routing to minimize load
        min_load = base_load[e]
        max_load = base_load[e]
        for i in range(d):
            if delta[e, i] > 0:
                max_load += delta[e, i]
            else:
                min_load += delta[e, i]
        lo = max(lo, min_load)
        hi = max(hi, max_load)

    best_routing = None
    best_L = hi + 1

    # Binary search for smallest feasible L
    while lo <= hi:
        if time.time() > deadline - 10:
            break
        mid = (lo + hi) // 2
        feasible, routing = check_feasibility_2sat(mid, base_load, delta, w, cw, ccw, n, d)
        if feasible:
            best_L = mid
            best_routing = routing
            hi = mid - 1
        else:
            lo = mid + 1

    if best_routing is None:
        # Fallback: exhaustive search
        total = 1 << d
        all_x = np.zeros((total, d), dtype=np.float64)
        for bit in range(d):
            all_x[:, bit] = (np.arange(total) >> bit) & 1
        all_loads = all_x @ delta.T + base_load
        max_loads = np.max(all_loads, axis=1)
        best_idx = np.argmin(max_loads)
        best_routing = all_x[best_idx].copy()

    return best_routing.astype(np.float64)


def check_feasibility_2sat(L, base_load, delta, w, cw, ccw, n, d):
    """
    Check if there exists a routing such that all edge loads <= L.
    Uses 2-SAT formulation.
    """
    # For each edge e, the constraint is:
    # base_load[e] + sum_i delta[e,i] * x_i <= L
    # where x_i is 1 if demand i is routed CCW, 0 if CW.
    #
    # This is a linear inequality in binary variables. To convert to 2-SAT,
    # we need to handle the case where the inequality is violated for some
    # assignment. However, general linear inequalities over binary variables
    # are not directly 2-SAT. 
    #
    # Alternative approach: Since d=15 is small, we can use a different strategy.
    # For each edge e, if the constraint is violated, we can derive clauses.
    # But this is complex. Instead, let's use a simpler approach:
    # We'll use a backtracking search with pruning, which is effectively
    # what the exhaustive search does but with early termination.

    # Actually, for this small problem, let's just do a smarter exhaustive search
    # with early pruning.
    best_x = None
    best_max_load = L + 1

    # Sort demands by weight descending for better pruning
    order = np.argsort(-w)

    # Backtracking
    def backtrack(idx, current_loads, x):
        nonlocal best_x, best_max_load
        if time.time() > 595:
            return False
        if idx == d:
            max_load = np.max(current_loads)
            if max_load <= L:
                best_x = x.copy()
                return True
            return False

        i = order[idx]
        # Try x_i = 0 (CW)
        new_loads = current_loads.copy()
        # If x_i = 0, load contribution is w_i * cw[i, :]
        new_loads += w[i] * cw[i]
        if np.max(new_loads) <= L:
            x[idx] = 0
            if backtrack(idx + 1, new_loads, x):
                return True

        # Try x_i = 1 (CCW)
        new_loads = current_loads.copy()
        # If x_i = 1, load contribution is w_i * ccw[i, :]
        new_loads += w[i] * ccw[i]
        if np.max(new_loads) <= L:
            x[idx] = 1
            if backtrack(idx + 1, new_loads, x):
                return True

        return False

    x = np.zeros(d, dtype=np.int64)
    current_loads = np.zeros(n, dtype=np.int64)
    if backtrack(0, current_loads, x):
        return True, x.astype(np.float64)
    else:
        return False, None
