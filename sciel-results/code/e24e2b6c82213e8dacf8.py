import numpy as np
import time

def ring_routing() -> np.ndarray:
    start_time = time.time()
    timeout = 590  # Safety margin before 600s
    
    n_nodes = 10
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
        (5, 2, 2),
    ]
    
    D = len(demands)
    
    # Precompute edge indices for CW and CCW
    # cw_edges[i] is a numpy array of edge indices
    # ccw_edges[i] is a numpy array of edge indices
    cw_idx = []
    ccw_idx = []
    for s, t, w in demands:
        cw = []
        if s == t:
            cw = list(range(n_nodes))
        else:
            edge = s
            while edge != t:
                cw.append(edge)
                edge = (edge + 1) % n_nodes
        cw_set = set(cw)
        ccw = [e for e in range(n_nodes) if e not in cw_set]
        
        cw_idx.append(np.array(cw, dtype=int))
        ccw_idx.append(np.array(ccw, dtype=int))
        
    # Precompute the load contribution vectors
    # contrib_cw[i] is a vector of length n_nodes with weight w on edges in cw_idx[i]
    # contrib_ccw[i] is a vector of length n_nodes with weight w on edges in ccw_idx[i]
    contrib_cw = []
    contrib_ccw = []
    for i, (s, t, w) in enumerate(demands):
        vec_cw = np.zeros(n_nodes, dtype=int)
        vec_ccw = np.zeros(n_nodes, dtype=int)
        if len(cw_idx[i]) > 0:
            vec_cw[cw_idx[i]] = w
        if len(ccw_idx[i]) > 0:
            vec_ccw[ccw_idx[i]] = w
        contrib_cw.append(vec_cw)
        contrib_ccw.append(vec_ccw)
        
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=float)
    
    # Iterate over all 2^D combinations
    # Using a simple bit counter for speed
    total_masks = 1 << D
    for mask in range(total_masks):
        # Check time
        if time.time() - start_time > timeout:
            break
            
        # Construct loads
        # We can do this efficiently by summing the chosen vectors
        # loads = sum(contrib_cw[i] if (mask >> i) & 1 == 0 else contrib_ccw[i] for i in range(D))
        # To speed up, we can accumulate in a numpy array
        loads = np.zeros(n_nodes, dtype=int)
        
        for i in range(D):
            if (mask >> i) & 1:
                loads += contrib_ccw[i]
            else:
                loads += contrib_cw[i]
                
        current_max = np.max(loads)
        
        if current_max < best_max_load:
            best_max_load = current_max
            # Decode mask to routing array
            # 0 = CW, 1 = CCW
            # The task says: "An entry >= 0.5 routes ... CCW, any other value CW"
            # So if bit is 1 (CCW), value should be >= 0.5 (e.g., 1.0)
            # If bit is 0 (CW), value should be < 0.5 (e.g., 0.0)
            routing = np.zeros(D, dtype=float)
            for i in range(D):
                if (mask >> i) & 1:
                    routing[i] = 1.0
                else:
                    routing[i] = 0.0
            best_routing = routing
            
    return best_routing
