import numpy as np
import time
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

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

    # Precompute CW and CCW edge index sets for each demand
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = []
        edge = s
        while edge != t:
            cw.append(edge)
            edge = (edge + 1) % n_nodes
        cw_edges.append(cw)
        ccw = [e for e in range(n_nodes) if e not in cw]
        ccw_edges.append(ccw)

    # Precompute the weight addition vectors for CW and CCW for each demand
    cw_vecs = []
    ccw_vecs = []
    for i in range(D):
        w = demands[i][2]
        cw_v = np.zeros(n_nodes)
        ccw_v = np.zeros(n_nodes)
        for e in cw_edges[i]:
            cw_v[e] = w
        for e in ccw_edges[i]:
            ccw_v[e] = w
        cw_vecs.append(cw_v)
        ccw_vecs.append(ccw_v)

    best_max_load = float('inf')
    best_routing = np.zeros(D)

    # Phase 1: Exhaustive search over all 2^15 = 32768 binary assignments
    for mask in range(1 << D):
        if time.time() > deadline:
            break

        loads = np.zeros(n_nodes)
        for i in range(D):
            if (mask >> i) & 1:
                # CCW (1)
                loads += ccw_vecs[i]
            else:
                # CW (0)
                loads += cw_vecs[i]

        max_load = np.max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = np.array([1.0 if ((mask >> i) & 1) else 0.0 for i in range(D)])

    # Phase 2: LP relaxation to find fractional solution
    # Decision variables: x_i in [0,1] for i in 0..D-1, where x_i=1 means CCW
    # Minimize t
    # Subject to: for each edge e, sum_i w_i * (x_i * ccw_vecs[i][e] + (1-x_i) * cw_vecs[i][e]) <= t
    #             0 <= x_i <= 1
    #
    # Load on edge e = sum_i w_i * cw_vecs[i][e] + sum_i w_i * (ccw_vecs[i][e] - cw_vecs[i][e]) * x_i
    #
    # Let base_e = sum_i w_i * cw_vecs[i][e]
    # Let delta_e_i = w_i * (ccw_vecs[i][e] - cw_vecs[i][e])
    #
    # Constraint: base_e + sum_i delta_e_i * x_i <= t
    #
    # Variables: x_0, ..., x_{D-1}, t
    # Objective: minimize t -> c = [0, ..., 0, 1]
    
    # Compute base loads (all CW)
    base_loads = np.zeros(n_nodes)
    for i in range(D):
        base_loads += cw_vecs[i]

    # Build constraint matrix A_ub and b_ub
    # For each edge e: base_e + sum_i delta_e_i * x_i - t <= 0
    # Variables order: x_0, ..., x_{D-1}, t
    A_ub = np.zeros((n_nodes, D + 1))
    b_ub = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        b_ub[e] = -base_loads[e]
        for i in range(D):
            delta = ccw_vecs[i][e] - cw_vecs[i][e]
            A_ub[e, i] = delta
        A_ub[e, D] = -1.0  # coefficient for t

    # Bounds: x_i in [0,1], t in [0, inf)
    bounds = [(0, 1)] * D + [(0, None)]

    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize t

    if time.time() < deadline:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            lp_max_load = res.fun
            if lp_max_load < best_max_load:
                # LP gives a fractional solution. We can use it to guide further search,
                # but for now, just record it. The integer optimum is >= LP optimum.
                pass

    return best_routing
