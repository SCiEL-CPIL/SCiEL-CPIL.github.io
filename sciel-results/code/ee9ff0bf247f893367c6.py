import time
import numpy as np
from itertools import combinations

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem via exhaustive vectorized search over
    all 2^15 = 32768 possible binary routing assignments, computing the
    maximum edge load for each and selecting the optimal one.
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
    cw = np.zeros((d, n), dtype=np.float64)
    ccw = np.zeros((d, n), dtype=np.float64)
    w = np.zeros(d, dtype=np.float64)

    for i, (s, t, wt) in enumerate(demands):
        w[i] = wt
        if s == t:
            continue
        e = s
        while e != t:
            cw[i, e] = 1.0
            e = (e + 1) % n
        ccw[i] = 1.0 - cw[i]

    # Edge load as function of routing x (0=CW, 1=CCW)
    # Load_e = base_e + sum_i w_i * (ccw[i,e] - cw[i,e]) * x_i
    base_load = np.dot(w, cw)  # shape (n,)
    delta = w[:, None] * (ccw - cw)  # shape (d, n)

    # Exhaustive search over all 2^d binary assignments
    # Generate all combinations as a (2^d, d) binary matrix
    total = 1 << d  # 32768
    all_x = np.zeros((total, d), dtype=np.float64)
    for bit in range(d):
        all_x[:, bit] = (np.arange(total) >> bit) & 1

    # Compute all edge loads: (total, d) @ (d, n) + base_load
    all_loads = all_x @ delta + base_load  # shape (total, n)
    
    # Find the maximum load for each assignment
    max_loads = np.max(all_loads, axis=1)  # shape (total,)
    
    # Find the best assignment
    best_idx = np.argmin(max_loads)
    best_score = max_loads[best_idx]
    best_x = all_x[best_idx].copy()

    # Verify time
    if time.time() > deadline:
        pass  # Already have the answer

    return best_x.astype(np.float64)
