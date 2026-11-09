# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges.
    Uses an exhaustive search over all 2^15 possible routing combinations to find
    the one that minimizes the maximum edge load.
    Uses Gray-code enumeration for incremental load updates.
    Optimized using Python lists for the inner loop to reduce overhead.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s limit

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

    weights = [w for _, _, w in demands]

    # Precompute delta vectors as Python lists for speed
    # delta[i][e] = weight_i if e in ccw_i else -weight_i if e in cw_i else 0
    # Wait, the logic in the original code was:
    # loads = sum(cw_matrix)
    # if switch to ccw: loads += delta where delta = ccw_matrix - cw_matrix
    # So delta[i][e] = w_i if e in ccw_i, -w_i if e in cw_i, 0 otherwise.
    
    delta = []
    for i in range(D):
        d = [0.0] * n_nodes
        w = weights[i]
        for e in cw_edges[i]:
            d[e] = -w
        for e in ccw_edges[i]:
            d[e] = w
        delta.append(d)

    # Initial loads: all demands routed CW
    loads = [0.0] * n_nodes
    for i in range(D):
        w = weights[i]
        for e in cw_edges[i]:
            loads[e] += w
    
    best_max_load = max(loads)
    best_mask = 0
    
    n_masks = 1 << D
    
    # Gray code enumeration
    for i in range(1, n_masks):
        if time.time() > deadline:
            break
        
        lsb = i & -i
        bit_pos = lsb.bit_length() - 1
        
        gray_i = i ^ (i >> 1)
        
        d = delta[bit_pos]
        if (gray_i >> bit_pos) & 1:
            # Switch to CCW: add delta
            for e in range(n_nodes):
                loads[e] += d[e]
        else:
            # Switch to CW: subtract delta
            for e in range(n_nodes):
                loads[e] -= d[e]
        
        current_max_load = max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = gray_i
    
    # Reconstruct best routing
    best_routing = np.array([(best_mask >> i) & 1 for i in range(D)], dtype=np.float64)
    return best_routing


# EVOLVE-BLOCK-END
