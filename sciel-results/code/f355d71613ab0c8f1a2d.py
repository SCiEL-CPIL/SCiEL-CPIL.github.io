# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def _compute_edge_contribs(n=10, demands=None):
    """
    For each demand, compute which edges are used if routed clockwise (CW)
    vs counter-clockwise (CCW).
    Returns (cw, ccw) arrays of shape (D, n).
    """
    if demands is None:
        demands = [
            (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
            (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
            (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
            (8, 1, 5), (0, 6, 1), (5, 2, 2),
        ]
    D = len(demands)
    cw = np.zeros((D, n))
    ccw = np.zeros((D, n))
    for i, (s, t, w) in enumerate(demands):
        # CW: edges s, s+1, ..., t-1 (mod n)
        if s <= t:
            edges = range(s, t)
        else:
            edges = list(range(s, n)) + list(range(0, t))
        for e in edges:
            cw[i, e] = w
        # CCW: all other edges
        for e in range(n):
            if cw[i, e] == 0:
                ccw[i, e] = w
    return cw, ccw


def _max_load(routing, cw, ccw):
    """
    routing: array of 0/1, 0=CW, 1=CCW
    Returns the maximum edge load.
    """
    loads = np.zeros(cw.shape[1])
    for i in range(len(routing)):
        if routing[i] >= 0.5:
            loads += ccw[i]
        else:
            loads += cw[i]
    return np.max(loads)


def ring_routing() -> np.ndarray:
    t_start = time.time()
    deadline = t_start + 620.0  # safety margin before 630s kill

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
        (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
        (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
        (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    cw, ccw = _compute_edge_contribs(n, demands)

    # Initial best: all CW
    best_routing = np.zeros(D)
    best_load = _max_load(best_routing, cw, ccw)

    # Also try all CCW
    alt = np.ones(D)
    alt_load = _max_load(alt, cw, ccw)
    if alt_load < best_load:
        best_routing = alt
        best_load = alt_load

    # LP formulation:
    # Variables: x_0..x_{D-1} (binary, 0=CW, 1=CCW), x_D (continuous, max load L)
    # Minimize: L
    # Subject to: for each edge e: sum_i (x_i * ccw[i,e] + (1-x_i) * cw[i,e]) <= L
    #   => sum_i x_i * (ccw[i,e] - cw[i,e]) + sum_i cw[i,e] - L <= 0
    #   => -sum_i x_i * (ccw[i,e] - cw[i,e]) + L >= sum_i cw[i,e]
    #   In standard form (<=): sum_i x_i * (ccw[i,e] - cw[i,e]) - L <= -sum_i cw[i,e]
    #
    # Wait, let me redo. Load on edge e = sum_i [x_i * ccw[i,e] + (1-x_i) * cw[i,e]]
    #   = sum_i x_i * ccw[i,e] + sum_i cw[i,e] - sum_i x_i * cw[i,e]
    #   = sum_i x_i * (ccw[i,e] - cw[i,e]) + sum_i cw[i,e]
    #
    # Constraint: sum_i x_i * (ccw[i,e] - cw[i,e]) + sum_i cw[i,e] <= L
    #   => sum_i x_i * (ccw[i,e] - cw[i,e]) - L <= -sum_i cw[i,e]

    # Objective: minimize L, so c has 0 for x_0..x_{D-1} and 1 for L
    c = np.zeros(D + 1)
    c[D] = 1.0

    # Inequality constraints A_ub @ x <= b_ub
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    for e in range(n):
        for i in range(D):
            A_ub[e, i] = ccw[i, e] - cw[i, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -np.sum(cw[:, e])

    # Bounds: x_i in [0,1] for i=0..D-1, L in [0, inf)
    bounds = [(0, 1)] * D + [(0, None)]

    # Solve LP with HiGHS
    lp_routing = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0 and res.x is not None:
            # Round binary variables
            x_rounded = np.where(res.x[:D] >= 0.5, 1.0, 0.0)
            lp_routing = x_rounded
            lp_load = _max_load(x_rounded, cw, ccw)
            if lp_load < best_load:
                best_routing = x_rounded
                best_load = lp_load
    except Exception:
        pass

    # Local search: single-bit flips
    improved = True
    while improved:
        improved = False
        for i in range(D):
            if time.time() > deadline:
                break
            candidate = best_routing.copy()
            candidate[i] = 1.0 - candidate[i]
            load = _max_load(candidate, cw, ccw)
            if load < best_load - 1e-12:
                best_routing = candidate
                best_load = load
                improved = True
        if time.time() > deadline:
            break

    # Pair flips
    improved = True
    while improved:
        improved = False
        for i in range(D):
            if time.time() > deadline:
                break
            for j in range(i + 1, D):
                if time.time() > deadline:
                    break
                candidate = best_routing.copy()
                candidate[i] = 1.0 - candidate[i]
                candidate[j] = 1.0 - candidate[j]
                load = _max_load(candidate, cw, ccw)
                if load < best_load - 1e-12:
                    best_routing = candidate
                    best_load = load
                    improved = True
            if time.time() > deadline:
                break

    return best_routing


# EVOLVE-BLOCK-END
