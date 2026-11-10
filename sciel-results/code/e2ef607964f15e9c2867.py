import numpy as np
import time
import itertools

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge masks as bitmasks or lists
    # Using lists of edge indices for simplicity, but we can use numpy for load calc
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        if s == t:
            cw_e = []
        else:
            cw_e = list(range(s, t)) if s < t else list(range(s, n)) + list(range(0, t))
        ccw_e = [e for e in range(n) if e not in cw_e]
        cw_edges.append(cw_e)
        ccw_edges.append(ccw_e)
        
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=float)
    
    start_time = time.time()
    
    # Iterate through all 2^D combinations
    # Using itertools.product is structurally different from Gray code
    for combo in itertools.product([0, 1], repeat=D):
        if time.time() - start_time > 590:
            break
            
        load = np.zeros(n, dtype=int)
        for i in range(D):
            w = demands[i][2]
            if combo[i] == 0:
                for e in cw_edges[i]:
                    load[e] += w
            else:
                for e in ccw_edges[i]:
                    load[e] += w
                    
        current_max = np.max(load)
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = np.array(combo, dtype=float)
            
    return best_routing
