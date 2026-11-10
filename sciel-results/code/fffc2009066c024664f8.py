import numpy as np
import time

def ring_routing() -> np.ndarray:
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
    
    # Precompute CW and CCW load vectors for each demand
    cw_loads = []
    ccw_loads = []
    
    for s, t, w in demands:
        # CW path from s to t: edges s, s+1, ..., t-1 (mod 10)
        cw_edges = []
        if s == t:
            cw_edges = []
        elif s < t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        
        cw_vec = np.zeros(n_nodes)
        for e in cw_edges:
            cw_vec[e] = w
            
        ccw_vec = np.zeros(n_nodes)
        for e in ccw_edges:
            ccw_vec[e] = w
            
        cw_loads.append(cw_vec)
        ccw_loads.append(ccw_vec)
    
    cw_arr = np.array(cw_loads)  # shape (15, 10)
    ccw_arr = np.array(ccw_loads)  # shape (15, 10)
    
    # base = sum of all CW loads
    # delta = ccw - cw for each demand
    base = np.sum(cw_arr, axis=0)  # shape (10,)
    deltas = ccw_arr - cw_arr  # shape (15, 10)
    
    start_time = time.time()
    timeout = 600.0
    safety_margin = 5.0
    
    best_max_load = float('inf')
    best_mask = 0
    
    # Exhaustive search over all 2^15 = 32768 configurations
    # Use vectorized approach for speed
    # Precompute prefix sums of deltas for faster load computation
    # Actually, let's use a smarter approach: iterate and maintain running load
    
    # For each mask, load = base + sum of deltas[i] for i where bit i is set
    # We can use Gray code to only change one bit at a time
    
    gray_code = [i ^ (i >> 1) for i in range(1 << n_demands)]
    prev_gray = 0
    current_load = base.copy()
    
    for idx in range(1 << n_demands):
        if (time.time() - start_time) > (timeout - safety_margin):
            break
        
        gray = gray_code[idx]
        
        # Find which bit changed from prev_gray to gray
        diff = prev_gray ^ gray
        if diff != 0:
            bit_pos = (diff & -diff).bit_length() - 1
            if gray & diff:
                # Bit was turned on
                current_load += deltas[bit_pos]
            else:
                # Bit was turned off
                current_load -= deltas[bit_pos]
        
        prev_gray = gray
        
        current_max = np.max(current_load)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = gray
    
    # Local search polish: try single flips and double flips from best
    # This is a refinement within the same design family
    
    def eval_mask(m):
        load = base.copy()
        for i in range(n_demands):
            if (m >> i) & 1:
                load += deltas[i]
        return np.max(load)
    
    # Single flip local search
    improved = True
    while improved:
        improved = False
        for i in range(n_demands):
            new_mask = best_mask ^ (1 << i)
            new_load = eval_mask(new_mask)
            if new_load < best_max_load:
                best_max_load = new_load
                best_mask = new_mask
                improved = True
        if (time.time() - start_time) > (timeout - safety_margin):
            break
    
    # Double flip local search
    improved = True
    while improved:
        improved = False
        for i in range(n_demands):
            for j in range(i + 1, n_demands):
                new_mask = best_mask ^ (1 << i) ^ (1 << j)
                new_load = eval_mask(new_mask)
                if new_load < best_max_load:
                    best_max_load = new_load
                    best_mask = new_mask
                    improved = True
        if (time.time() - start_time) > (timeout - safety_margin):
            break
    
    # Triple flip local search for more thorough polish
    improved = True
    while improved:
        improved = False
        for i in range(n_demands):
            for j in range(i + 1, n_demands):
                for k in range(j + 1, n_demands):
                    new_mask = best_mask ^ (1 << i) ^ (1 << j) ^ (1 << k)
                    new_load = eval_mask(new_mask)
                    if new_load < best_max_load:
                        best_max_load = new_load
                        best_mask = new_mask
                        improved = True
        if (time.time() - start_time) > (timeout - safety_margin):
            break
    
    # Construct result
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0  # CCW
        else:
            routing[i] = 0.0  # CW
            
    return routing
