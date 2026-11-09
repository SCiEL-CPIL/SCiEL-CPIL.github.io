# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges.
    Uses an exhaustive search over all 2^15 possible routing combinations to find
    the one that minimizes the maximum edge load.
    """
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

    # Precompute the edge sets for clockwise and counter-clockwise routes
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        # Clockwise: edges s, s+1, ..., t-1 (mod n)
        cw = set()
        e = s
        while e != t:
            cw.add(e)
            e = (e + 1) % n_nodes
        cw_edges.append(cw)
        # Counter-clockwise: all other edges
        all_edges = set(range(n_nodes))
        ccw = all_edges - cw
        ccw_edges.append(ccw)

    # Convert to boolean matrices for fast vectorized operations
    # cw_matrix[i][j] = 1 if demand i uses edge j when routed clockwise
    cw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    ccw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    for i in range(D):
        for e in cw_edges[i]:
            cw_matrix[i, e] = 1.0
        for e in ccw_edges[i]:
            ccw_matrix[i, e] = 1.0

    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    # Precompute base loads and diff matrix
    base_loads = weights @ cw_matrix  # loads if all clockwise
    diff_matrix = ccw_matrix - cw_matrix  # change in load per edge when switching to CCW
    diff_T = diff_matrix.T  # (n_nodes, D)

    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)

    # Exhaustive search over all 2^15 combinations
    for mask in range(1 << D):
        # routing[i] = 1 if counter-clockwise, 0 if clockwise
        routing = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.float64)

        # Compute edge loads using precomputed matrices
        switch_loads = weights * routing  # weighted switch vector
        loads = base_loads + diff_T @ switch_loads

        max_load = np.max(loads)

        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = routing.copy()

    return best_routing


# EVOLVE-BLOCK-END
