import numpy as np
import time

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin
    
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
        (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute edge usage for CW and CCW
    cw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    ccw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_vec[i, current] = 1
            current = (current + 1) % n_nodes
        ccw_vec[i] = 1 - cw_vec[i]
        
    weights = np.array([w for s, t, w in demands], dtype=np.float64)
    
    L_cw = cw_vec.T @ weights
    Delta = weights[:, np.newaxis] * (ccw_vec - cw_vec)
    
    # Generate all 2^15 masks
    masks = np.arange(1 << n_demands)
    D = (masks[:, np.newaxis] >> np.arange(n_demands)) & 1
    
    # Calculate loads for all combinations
    loads_all = L_cw[None, :] + D @ Delta
    
    # Find the combination with minimum max load
    max_loads = np.max(loads_all, axis=1)
    best_idx = np.argmin(max_loads)
    
    best_routing = D[best_idx].astype(np.float64)
    
    # Deadline check (though this should be very fast)
    if time.time() > deadline - 1.0:
         # Return best found so far (which is the only one)
         pass
         
    return best_routing
