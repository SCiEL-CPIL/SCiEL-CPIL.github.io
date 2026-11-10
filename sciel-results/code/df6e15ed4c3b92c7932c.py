import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute CW and CCW edge incidence matrices
    # A_cw[i, e] = weight of demand i if routed CW and it uses edge e
    # A_ccw[i, e] = weight of demand i if routed CCW and it uses edge e
    A_cw = np.zeros((D, n), dtype=int)
    A_ccw = np.zeros((D, n), dtype=int)
    
    for i, (s, t, w) in enumerate(demands):
        if s < t:
            edges = range(s, t)
        else:
            edges = list(range(s, n)) + list(range(0, t))
        
        for e in edges:
            A_cw[i, e] = w
            # CCW uses the complement
            # We can just fill the complement later or calculate directly
            # Direct calculation for CCW:
            # Edges are t to s (wrap around)
            if t < s:
                ccw_edges = range(t, s)
            else:
                ccw_edges = list(range(t, n)) + list(range(0, s))
            for e in ccw_edges:
                A_ccw[i, e] = w

    # Total weight vector
    w_vec = np.array([d[2] for d in demands])
    
    # Load if all CW
    load_all_cw = A_cw.sum(axis=0)
    # Load if all CCW
    load_all_ccw = A_ccw.sum(axis=0)
    
    # Delta load for switching demand i from CW to CCW
    # load = load_all_cw + sum( (A_ccw[i] - A_cw[i]) * x_i )
    # Let D_i = A_ccw[i] - A_cw[i]
    D_mat = A_ccw - A_cw
    
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=float)
    
    start_time = time.time()
    
    # Iterate through all 2^D combinations
    # Using a simple integer counter for speed
    for mask in range(1 << D):
        if time.time() - start_time > 590:
            break
            
        # Construct x vector from mask
        # x_i = 1 if bit i is set (CCW), else 0 (CW)
        # We can use numpy to get bits
        # But for D=15, a python loop to build the vector might be slower than the vectorized load calc?
        # Actually, we can precompute the load for all masks? No, 32k masks x 10 edges is small.
        
        # Efficient way:
        # load = load_all_cw + D_mat @ x
        # x is a vector of 0s and 1s.
        
        # Get x from mask
        # x = np.array([(mask >> i) & 1 for i in range(D)], dtype=int)
        # This is a python loop.
        
        # Alternative:
        # Precompute loads for all masks?
        # We can do this recursively or using Gray code?
        # For D=15, 32768 iterations.
        # Inside:
        # x = ...
        # load = load_all_cw + D_mat.dot(x)
        # max_load = np.max(load)
        
        # Let's try to vectorize the mask expansion.
        # Not easily for arbitrary mask in a single numpy call without precomputation.
        
        # However, 32768 * (15 bit extractions + dot product) is fast.
        
        x = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.int8)
        load = load_all_cw + D_mat.dot(x)
        current_max = np.max(load)
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = x.astype(float)
            
    return best_routing
