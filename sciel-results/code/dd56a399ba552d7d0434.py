# EVOLVE-BLOCK-START
import numpy as np
from itertools import product


def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge sets for each demand in both directions
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
    # Counter-clockwise: all other edges
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        if s == t:
            cw_e = []
        else:
            cw_e = list(range(s, t)) if s < t else list(range(s, n)) + list(range(0, t))
        ccw_e = [e for e in range(n) if e not in cw_e]
        cw_edges.append(cw_e)
        ccw_edges.append(ccw_e)
    
    # Enumerate all 2^15 routing combinations
    best_routing = None
    best_max_load = float('inf')
    
    for bits in product([0, 1], repeat=D):
        # bits[i] = 0 -> clockwise, bits[i] = 1 -> counter-clockwise
        load = [0] * n
        for i in range(D):
            s, t, w = demands[i]
            if bits[i] == 0:
                for e in cw_edges[i]:
                    load[e] += w
            else:
                for e in ccw_edges[i]:
                    load[e] += w
        max_load = max(load)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = bits
    
    # Convert to output format: >= 0.5 means counter-clockwise
    routing = np.zeros(D)
    for i in range(D):
        if best_routing[i] == 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
    
    return routing


# EVOLVE-BLOCK-END
