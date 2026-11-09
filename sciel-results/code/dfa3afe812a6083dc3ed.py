import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    
    # Precompute edge masks
    mask_cw = np.zeros((D, n), dtype=np.int32)
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            mask_cw[d, e % n] = 1
            e = (e + 1) % n
            
    weights = np.array([w for _, _, w in demands], dtype=np.float64)
    
    # base_cw_load: load on each edge if all routed CW
    base_cw_load = (weights[:, None] * mask_cw).sum(axis=0)
    
    # delta: if we flip demand d to CCW, the change in load on each edge
    # CCW uses edges NOT in CW. So contribution changes from w*mask_cw to w*(1-mask_cw)
    # delta[d] = w*(1-mask_cw) - w*mask_cw = w*(1-2*mask_cw)
    delta = weights[:, None] * (1 - 2 * mask_cw)
    
    start_time = time.time()
    deadline = start_time + 595.0  # 595s out of 630s, leaving 35s safety margin
    
    # Exhaustive search: 2^15 = 32768 combinations, trivially fast
    # Vectorized approach:
    # For each of the 2^D combinations, compute loads = base_cw_load + sum of delta[d] for each d routed CCW
    
    # Build all combinations as bitmasks
    # For each bitmask, we need to sum delta[d] for each bit set
    # This is equivalent to: loads = base_cw_load + @ (delta, bitmask_as_vector)
    
    # Efficient approach: iterate through all 2^D masks and use bit operations
    # Or: precompute all subsets via DP
    
    # Method: for each edge e, load[e] = base_cw_load[e] + sum_{d: bit d set} delta[d,e]
    # We can compute this for all masks at once using a clever trick:
    # mask_matrix[m, d] = 1 if bit d is set in m, else 0
    # Then loads[m, e] = base_cw_load[e] + sum_d mask_matrix[m, d] * delta[d, e]
    
    total_masks = 1 << D  # 32768
    
    # Build mask_matrix: shape (32768, 15)
    # mask_matrix[m, d] = (m >> d) & 1
    all_masks = np.arange(total_masks, dtype=np.uint32)
    mask_matrix = np.zeros((total_masks, D), dtype=np.float64)
    for d in range(D):
        mask_matrix[:, d] = ((all_masks >> d) & 1).astype(np.float64)
    
    # loads = base_cw_load[None, :] + mask_matrix @ delta
    # mask_matrix @ delta has shape (32768, 10)
    loads = base_cw_load[None, :] + mask_matrix @ delta
    
    # max_load for each combination
    max_loads = loads.max(axis=1)
    
    # Find the combination with minimum max_load
    best_idx = np.argmin(max_loads)
    best_max_load = max_loads[best_idx]
    
    # Extract the routing for the best combination
    best_mask = all_masks[best_idx]
    best_x = np.zeros(D, dtype=np.float64)
    for d in range(D):
        if (best_mask >> d) & 1:
            best_x[d] = 1.0  # CCW
        else:
            best_x[d] = 0.0  # CW
    
    # Safety check: ensure we haven't exceeded time
    if time.time() > deadline:
        # This shouldn't happen for 2^15, but just in case
        pass
    
    return best_x
