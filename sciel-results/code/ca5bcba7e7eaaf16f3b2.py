import numpy as np
import time

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    
    # Precompute contribution vectors for CW and CCW routing
    # cw_contrib[d] is an array of length n, where cw_contrib[d][e] is the weight if demand d is routed CW
    # ccw_contrib[d] is an array of length n, where ccw_contrib[d][e] is the weight if demand d is routed CCW
    cw_contrib = np.zeros((D, n), dtype=float)
    ccw_contrib = np.zeros((D, n), dtype=float)
    
    for d, (s, t, w) in enumerate(demands):
        # Determine edges used in CW direction: s, s+1, ..., t-1
        cw_edges = set()
        e = s
        while e != t:
            cw_edges.add(e % n)
            e = (e + 1) % n
            
        # CCW uses all other edges
        for e in range(n):
            if e in cw_edges:
                cw_contrib[d, e] = w
                # ccw_contrib[d, e] remains 0
            else:
                ccw_contrib[d, e] = w
                # cw_contrib[d, e] remains 0

    # Exhaustive search over all 2^D combinations
    best_max_load = float('inf')
    best_mask = 0
    
    # Total combinations is 2^15 = 32768
    # We iterate through all possible routing assignments
    for mask in range(1 << D):
        # Check deadline
        if time.time() > deadline - 10:
            break
            
        # Calculate total load
        # load[e] = sum( w_d * I(direction_d) )
        # We can sum the vectors
        # To optimize, we can accumulate the load vector
        load = np.zeros(n, dtype=float)
        
        for d in range(D):
            if (mask >> d) & 1:
                # CCW
                load += ccw_contrib[d]
            else:
                # CW
                load += cw_contrib[d]
                
        current_max_load = np.max(load)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = mask
            
            # Early termination if we reach the theoretical lower bound?
            # The lower bound is at least max(demand weight) = 5?
            # Actually, the sum of weights is 3+2+4+1+5+2+3+2+4+1+3+2+5+1+2 = 40.
            # Average load per edge is 40/10 = 4.
            # So max load must be at least 4.
            # If we find 4, it's optimal.
            if best_max_load <= 4.0:
                break

    # Construct the result array
    # 1.0 for CCW (bit is 1), 0.0 for CW (bit is 0)
    result = np.zeros(D, dtype=float)
    for d in range(D):
        if (best_mask >> d) & 1:
            result[d] = 1.0
        else:
            result[d] = 0.0
            
    return result
