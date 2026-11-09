# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring loading problem for 10 nodes and 15 demands.
    Uses exhaustive search over all 2^15 routing combinations to guarantee the global optimum.
    Optimized to process all combinations in a single vectorized step.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
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

    # Precompute clockwise edge-usage matrix
    # cw[d, e] = 1 if demand d uses edge e clockwise
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

    # Counter-clockwise uses the complementary edges
    ccw = 1.0 - cw

    # Load calculation:
    # load[e] = sum_d weight[d] * (x[d] * ccw[d,e] + (1-x[d]) * cw[d,e])
    #         = sum_d weight[d] * cw[d,e] + sum_d weight[d] * x[d] * (ccw[d,e] - cw[d,e])
    base_load = cw.T @ weights  # Load if all demands are routed clockwise
    delta = ccw - cw            # Change in edge usage when switching d from CW to CCW
    w_delta = delta * weights[:, np.newaxis]

    total_combos = 1 << D
    
    # Since 2^15 = 32768 is small, we can process all combinations at once.
    # This eliminates loop overhead and allows for a single vectorized operation.
    indices = np.arange(total_combos, dtype=np.int64)
    
    # Extract bits: x_d = (index >> d) & 1
    # We need bits shape (total_combos, D)
    shifts = np.arange(D, dtype=np.int64)
    bits = ((indices[:, np.newaxis] >> shifts[np.newaxis, :]) & 1).astype(np.float64)
    
    # Compute loads for all combinations
    # loads shape: (total_combos, n)
    # loads = base_load + bits @ w_delta
    loads = base_load[np.newaxis, :] + bits @ w_delta
    
    # Find the combination with the minimum maximum load
    max_loads = loads.max(axis=1)
    best_idx = np.argmin(max_loads)
    best_routing = bits[best_idx].copy()
    
    # Check deadline - if we're close to the limit, return immediately
    elapsed = time.time() - start_time
    if elapsed > deadline - 30.0:  # 30s safety margin
        return best_routing
    
    # The exhaustive search is already complete and optimal.
    # No further improvement is possible for this exact problem size.
    return best_routing

# EVOLVE-BLOCK-END
