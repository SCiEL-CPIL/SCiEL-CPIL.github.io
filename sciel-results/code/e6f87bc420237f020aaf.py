import numpy as np
import time

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5), 
        (2,8,2), (5,9,3), (1,4,2), (6,0,4), (3,8,1), 
        (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    
    deadline = time.time() + 600.0
    
    # Precompute edge lists
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        weights.append(w)
        # CW: s -> t
        edges = []
        e = s
        while e != t:
            edges.append(e)
            e = (e + 1) % N
        cw_edges.append(edges)
        
        # CCW: t -> s
        edges = []
        e = t
        while e != s:
            edges.append(e)
            e = (e + 1) % N
        ccw_edges.append(edges)
    
    # Use numpy arrays for faster load computation
    cw_arr = np.zeros((D, N), dtype=np.float64)
    ccw_arr = np.zeros((D, N), dtype=np.float64)
    for i in range(D):
        for e in cw_edges[i]:
            cw_arr[i, e] = 1.0
        for e in ccw_edges[i]:
            ccw_arr[i, e] = 1.0
    w_arr = np.array(weights, dtype=np.float64)
    
    best_max_load = float('inf')
    best_assignment = None
    
    # Precompute CW and CCW load contributions for each demand
    # cw_contrib[i] = w_i * cw_arr[i], ccw_contrib[i] = w_i * ccw_arr[i]
    cw_contrib = cw_arr * w_arr[:, None]  # (D, N)
    ccw_contrib = ccw_arr * w_arr[:, None]  # (D, N)
    
    # Brute force 2^15 = 32768 combinations
    for mask in range(1 << D):
        if time.time() > deadline - 10.0:
            break
        
        # Compute loads: for each demand, pick CW or CCW based on bit
        # loads = sum over i of (bit_i * ccw_contrib[i] + (1-bit_i) * cw_contrib[i])
        # Equivalent to: loads = sum(cw_contrib) + sum over i where bit_i=1 of (ccw_contrib[i] - cw_contrib[i])
        # Precompute base = sum(cw_contrib) and delta[i] = ccw_contrib[i] - cw_contrib[i]
        bits = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.float64)
        # loads = base_load + sum of delta[i] for i where bit_i=1
        delta = ccw_contrib - cw_contrib  # (D, N)
        loads = np.sum(cw_contrib, axis=0) + delta.T @ bits
        
        current_max = np.max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_assignment = mask
            
    # Polish: verify best and check single-bit flips to ensure local optimality
    if best_assignment is not None:
        # Compute loads for best_assignment
        bits_best = np.array([(best_assignment >> i) & 1 for i in range(D)], dtype=np.float64)
        delta = ccw_contrib - cw_contrib
        base_load = np.sum(cw_contrib, axis=0)
        loads_best = base_load + delta.T @ bits_best
        current_max = np.max(loads_best)
        
        # Try flipping each single bit to see if we can improve
        for i in range(D):
            if time.time() > deadline - 5.0:
                break
            new_mask = best_assignment ^ (1 << i)
            bits_new = np.array([(new_mask >> j) & 1 for j in range(D)], dtype=np.float64)
            loads_new = base_load + delta.T @ bits_new
            new_max = np.max(loads_new)
            if new_max < current_max:
                best_assignment = new_mask
                current_max = new_max
                bits_best = bits_new
                loads_best = loads_new
    
    # Convert best_assignment to output array
    # Entry >= 0.5 -> CCW (1), < 0.5 -> CW (0)
    routing = np.zeros(D)
    for i in range(D):
        if (best_assignment >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
