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
    cw_mask = np.zeros((n_demands, n_nodes), dtype=np.float64)
    ccw_mask = np.zeros((n_demands, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_mask[i, current] = 1.0
            current = (current + 1) % n_nodes
        ccw_mask[i] = 1.0 - cw_mask[i]
    
    weights = np.array([w for _, _, w in demands])
    cw_weighted = cw_mask * weights[:, np.newaxis]
    ccw_weighted = ccw_mask * weights[:, np.newaxis]
    
    base_load = cw_weighted.sum(axis=0)
    delta = ccw_weighted - cw_weighted  # [n_demands, n_nodes]
    
    # Brute force over all 2^15 combinations using Gray code for O(1) update per step
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    n_total = 1 << n_demands
    loads = base_load.copy()
    prev_gray = 0
    
    for idx in range(n_total):
        if time.time() > deadline:
            break
        
        # Gray code: gray = idx ^ (idx >> 1)
        gray = idx ^ (idx >> 1)
        
        if idx > 0:
            # Find which bit changed between prev_gray and gray
            diff = gray ^ prev_gray
            bit_pos = diff.bit_length() - 1
            # If bit is now set, add delta[bit_pos]; else subtract
            if gray & diff:
                loads += delta[bit_pos]
            else:
                loads -= delta[bit_pos]
        
        prev_gray = gray
        
        current_max = loads.max()
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = gray  # store the gray-code index, but we need the actual mask
    
    # best_mask is actually the gray code value; convert back to binary mask
    # gray = mask ^ (mask >> 1), so mask = ?
    # We need to recover the original binary mask from the gray code
    # Actually, let's track the actual mask differently
    
    # Let me redo: track actual mask alongside
    # Re-do with proper tracking
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    loads = base_load.copy()
    prev_gray = 0
    prev_mask = 0
    
    for idx in range(n_total):
        if time.time() > deadline:
            break
        
        gray = idx ^ (idx >> 1)
        
        if idx > 0:
            diff = gray ^ prev_gray
            bit_pos = diff.bit_length() - 1
            if gray & diff:
                loads += delta[bit_pos]
            else:
                loads -= delta[bit_pos]
        
        prev_gray = gray
        prev_mask = gray  # We'll use gray as our index, but need to map back
        
        current_max = loads.max()
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = gray
    
    # Convert gray code to binary: binary = 0; while gray: binary ^= gray; gray >>= 1
    # Actually: to convert gray to binary:
    # binary = gray
    # while gray >>= 1: binary ^= gray
    gray_val = best_mask
    binary_mask = gray_val
    g = gray_val
    g >>= 1
    while g:
        binary_mask ^= g
        g >>= 1
    
    # Build result: 1.0 for CCW (>= 0.5), 0.0 for CW
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (binary_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
    
    return result
