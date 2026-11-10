import numpy as np
import time

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s limit

    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]

    # Precompute edge usage matrices
    cw_mat = np.zeros((D, n), dtype=np.float64)
    ccw_mat = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        d_cw = (t - s) % n
        if d_cw > 0:
            for k in range(d_cw):
                e = (s + k) % n
                cw_mat[i, e] = 1.0
            for e in range(n):
                if cw_mat[i, e] == 0.0:
                    ccw_mat[i, e] = 1.0

    # base_load: load on each edge if all demands go CW
    base_load = weights @ cw_mat  # shape (n,)
    
    # delta_mat[i] = w_i * (ccw_mat[i] - cw_mat[i])
    # When demand i is routed CCW (x_i=1), the load change on each edge is delta_mat[i]
    delta_mat = (ccw_mat - cw_mat) * weights[:, None]

    # Exhaustive search over all 2^15 = 32768 combinations
    best_val = np.inf
    best_mask = 0
    
    for mask in range(1 << D):
        load = base_load.copy()
        for i in range(D):
            if mask & (1 << i):
                load += delta_mat[i]
        
        max_load = np.max(load)
        if max_load < best_val - 1e-9:
            best_val = max_load
            best_mask = mask
    
    # Deadline check (should never trigger for 32768 iterations)
    if time.time() > deadline:
        # Return best found so far
        pass

    # Reconstruct solution
    best_x = np.array([(best_mask >> i) & 1 for i in range(D)], dtype=np.float64)
    return best_x
