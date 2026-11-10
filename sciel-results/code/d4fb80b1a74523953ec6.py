# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 (>= 0.5) -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s budget with safety margin

    n = 10  # number of nodes/edges
    D = 15  # number of demands

    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute for each demand:
    # - CW edges: edges from s to t going clockwise (s, s+1, ..., t-1 mod n)
    # - CCW edges: all other edges
    cw_edge_masks = []
    ccw_edge_masks = []

    for s, t, w in demands:
        # CW edges: s, (s+1)%n, ..., (t-1)%n
        cw_edges = set()
        e = s
        while e != t:
            cw_edges.add(e)
            e = (e + 1) % n
        cw_mask = np.zeros(n, dtype=np.int32)
        for e in cw_edges:
            cw_mask[e] = 1
        cw_edge_masks.append(cw_mask)

        # CCW edges: all edges not in CW
        ccw_mask = 1 - cw_mask
        ccw_edge_masks.append(ccw_mask)

    # Weights array
    weights = np.array([w for s, t, w in demands], dtype=np.float64)

    # Exhaustive search over all 2^15 = 32768 combinations
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)

    # For each combination, compute the load on each edge
    # Using bitmask representation: bit i = 1 means demand i is CCW
    total_combos = 1 << D

    # Precompute contribution of each demand to each edge in both directions
    # cw_contrib[i, e] = weight_i if demand i uses edge e when CW, else 0
    # ccw_contrib[i, e] = weight_i if demand i uses edge e when CCW, else 0
    cw_contrib = np.zeros((D, n), dtype=np.float64)
    ccw_contrib = np.zeros((D, n), dtype=np.float64)
    for i in range(D):
        cw_contrib[i] = weights[i] * cw_edge_masks[i]
        ccw_contrib[i] = weights[i] * ccw_edge_masks[i]

    # For efficiency, iterate over all 2^15 combinations
    for combo in range(total_combos):
        # Check deadline
        if time.time() > deadline - 5.0:
            break

        # Compute edge loads
        # For each demand i, if bit i is set, use CCW, else CW
        loads = np.zeros(n, dtype=np.float64)
        for i in range(D):
            if (combo >> i) & 1:
                loads += ccw_contrib[i]
            else:
                loads += cw_contrib[i]

        max_load = np.max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = np.zeros(D, dtype=np.float64)
            for i in range(D):
                if (combo >> i) & 1:
                    best_routing[i] = 1.0
                else:
                    best_routing[i] = 0.0

    return best_routing


# EVOLVE-BLOCK-END
