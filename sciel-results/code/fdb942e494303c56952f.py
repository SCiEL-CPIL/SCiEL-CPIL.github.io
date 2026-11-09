import numpy as np

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5), 
        (2,8,2), (5,9,3), (1,4,2), (6,0,4), (3,8,1), 
        (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    
    # Precompute contribution matrices
    # cw_contrib[i, e] = weight_i if edge e is in CW path of demand i, else 0
    # ccw_contrib[i, e] = weight_i if edge e is in CCW path of demand i, else 0
    cw_contrib = np.zeros((D, N), dtype=np.int64)
    ccw_contrib = np.zeros((D, N), dtype=np.int64)
    
    for i, (s, t, w) in enumerate(demands):
        # CW path: s -> s+1 -> ... -> t
        e = s
        while e != t:
            cw_contrib[i, e] = w
            e = (e + 1) % N
            
        # CCW path: complement of CW path
        # Equivalent to path t -> t+1 -> ... -> s
        e = t
        while e != s:
            ccw_contrib[i, e] = w
            e = (e + 1) % N
            
    delta_contrib = ccw_contrib - cw_contrib
    base_load = np.sum(cw_contrib, axis=0)
    
    best_max_load = np.inf
    best_mask = 0
    
    # Gray code iteration
    # Start with mask 0 (all CW)
    current_loads = base_load.copy()
    current_max = np.max(current_loads)
    if current_max < best_max_load:
        best_max_load = current_max
        best_mask = 0
        
    prev_gray = 0
    for i in range(1, 1 << D):
        gray = i ^ (i >> 1)
        
        # Find the bit that changed
        diff = gray ^ prev_gray
        # diff is a power of 2, so log2(diff) gives the bit index
        # Or use bit_length - 1
        changed_bit = diff.bit_length() - 1
        
        # Update loads
        if gray & diff:
            # Bit turned ON (CW -> CCW)
            current_loads += delta_contrib[changed_bit]
        else:
            # Bit turned OFF (CCW -> CW)
            current_loads -= delta_contrib[changed_bit]
            
        current_max = np.max(current_loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = i
            
        prev_gray = gray
        
    # Construct result
    routing = np.zeros(D, dtype=np.float64)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
