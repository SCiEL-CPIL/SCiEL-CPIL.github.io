import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search over all 2^15 routing 
    combinations to find the one that minimizes the maximum edge load.
    """
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
    
    # Pre-compute edge indicator vectors
    cw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    ccw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_vec[i, current] = 1
            current = (current + 1) % n_nodes
        ccw_vec[i] = 1 - cw_vec[i]
    
    weights = np.array([w for s, t, w in demands], dtype=np.int32)
    diff = ccw_vec - cw_vec  # +1 for CCW-only edges, -1 for CW-only edges
    
    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    for mask in range(1 << n_demands):
        if time.time() > deadline:
            break
        
        directions = np.array([(mask >> i) & 1 for i in range(n_demands)], dtype=np.int8)
        used = cw_vec + directions[:, np.newaxis] * diff
        loads = used.T @ weights
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_routing = directions.astype(np.float64)
    
    return best_routing
