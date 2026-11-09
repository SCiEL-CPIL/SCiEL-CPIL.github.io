import numpy as np
import time

def ring_routing() -> np.ndarray:
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute for each demand: which edges are used in CW and CCW
    # cw_mask[i, e] = 1 if edge e is on CW path for demand i
    # ccw_mask[i, e] = 1 if edge e is on CCW path for demand i
    cw_mask = np.zeros((n_demands, n_nodes), dtype=np.float64)
    ccw_mask = np.zeros((n_demands, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # CW edges: s, s+1, ..., t-1 (mod n_nodes)
        current = s
        while current != t:
            cw_mask[i, current] = 1.0
            current = (current + 1) % n_nodes
        # CCW: complement
        ccw_mask[i] = 1.0 - cw_mask[i]
    
    # Weighted contributions
    weights = np.array([w for _, _, w in demands])
    cw_weighted = cw_mask * weights[:, np.newaxis]   # [n_demands, n_nodes]
    ccw_weighted = ccw_mask * weights[:, np.newaxis]  # [n_demands, n_nodes]
    
    # base_load[e] = load on edge e if all demands go CW
    base_load = cw_weighted.sum(axis=0)  # [n_nodes]
    
    # delta[i, e] = change in load on edge e if demand i switches from CW to CCW
    delta = ccw_weighted - cw_weighted  # [n_demands, n_nodes]
    
    # Brute force over all 2^15 combinations
    # For a given mask (bit i = 1 means demand i goes CCW):
    # loads = base_load + sum over i where bit i=1 of delta[i]
    
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Efficient iteration using numpy
    for mask in range(1 << n_demands):
        if time.time() > deadline:
            break
        
        # Compute loads for this mask
        # Select rows where bit is set
        # For efficiency, use a lookup approach
        m = mask
        loads = base_load.copy()
        i = 0
        while m:
            if m & 1:
                loads += delta[i]
            m >>= 1
            i += 1
        
        current_max = loads.max()
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask
    
    # Build result: 1.0 for CCW (>= 0.5), 0.0 for CW
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
    
    return result
