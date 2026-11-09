# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of 10 nodes with 15 demands.
    Exhaustive search over all 2^15 routing combinations, vectorized with NumPy.
    """
    deadline = time.time() + 580.0  # safety margin before 630s kill

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
    D = len(demands)
    weights = np.array([d[2] for d in demands], dtype=np.float64)

    # Precompute clockwise edge-usage matrix: cw[d, e] = 1 if demand d uses edge e clockwise
    # Edge e connects node e to node (e+1) mod 10
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
    cw = np.zeros((D, n), dtype=np.float64)
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            for e in range(s, t):
                cw[i, e] = 1.0
        else:
            for e in range(s, n):
                cw[i, e] = 1.0
            for e in range(0, t):
                cw[i, e] = 1.0

    # Counter-clockwise uses all other edges: ccw = 1 - cw (but only for edges not on CW path)
    # Actually, the ring has n edges total. CW path uses some edges, CCW uses the rest.
    ccw = 1.0 - cw

    # For a routing choice x (binary vector), the load on edge e is:
    #   load[e] = sum over demands d: weight[d] * (x[d] * ccw[d,e] + (1-x[d]) * cw[d,e])
    #           = sum_d weight[d] * cw[d,e] + sum_d weight[d] * x[d] * (ccw[d,e] - cw[d,e])
    # Let base_load[e] = sum_d weight[d] * cw[d,e]  (all clockwise)
    # Let delta[d,e] = ccw[d,e] - cw[d,e]
    # load[e] = base_load[e] + sum_d weight[d] * x[d] * delta[d,e]

    base_load = cw.T @ weights  # shape (n,)
    delta = ccw - cw  # shape (D, n), values in {-1, 0, 1}
    w_delta = delta * weights[:, np.newaxis]  # shape (D, n)

    # Now enumerate all 2^15 combinations
    # Process in chunks to keep memory bounded
    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.float64)

    total_combos = 1 << D
    chunk_size = 4096

    for start in range(0, total_combos, chunk_size):
        if time.time() > deadline:
            break
        end = min(start + chunk_size, total_combos)
        size = end - start
        indices = np.arange(start, end, dtype=np.int64)

        # For each combination, get the bit vector x (D bits)
        # x_d = (index >> d) & 1
        bits = ((indices[:, np.newaxis] >> np.arange(D)[np.newaxis, :]) & 1).astype(np.float64)
        # bits shape: (size, D)

        # Compute loads: load[e] = base_load[e] + sum_d bits[d] * w_delta[d, e]
        # loads shape: (size, n)
        loads = base_load[np.newaxis, :] + bits @ w_delta

        # Max load per combination
        max_loads = loads.max(axis=1)

        # Find best in this chunk
        local_best_idx = np.argmin(max_loads)
        if max_loads[local_best_idx] < best_max_load:
            best_max_load = max_loads[local_best_idx]
            best_routing = bits[local_best_idx].copy()

    return best_routing


# EVOLVE-BLOCK-END
