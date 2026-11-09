import numpy as np
import time

def ring_routing() -> np.ndarray:
    # Define demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute edge indices for CW and CCW routing for each demand
    # CW: s -> s+1 -> ... -> t (mod n)
    # CCW: s -> s-1 -> ... -> t (mod n)
    
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        # CW edges
        e_cw = []
        curr = s
        while curr != t:
            e_cw.append(curr)
            curr = (curr + 1) % n_nodes
        cw_edges.append(e_cw)
        
        # CCW edges
        e_ccw = []
        curr = s
        while curr != t:
            curr = (curr - 1) % n_nodes
            e_ccw.append(curr)
        ccw_edges.append(e_ccw)
        
    # Convert to numpy arrays for faster indexing if needed, but lists are fine for small N
    # We will iterate over all 2^15 combinations
    
    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600 # 600 seconds
    
    # Iterate over all 2^15 possibilities
    for mask in range(1 << n_demands):
        if time.time() > deadline - 10: # Safety margin
            break
            
        loads = np.zeros(n_nodes)
        routing = np.zeros(n_demands)
        
        for i in range(n_demands):
            if (mask >> i) & 1:
                # CCW
                routing[i] = 1.0
                # Add weight to CCW edges
                # Using numpy addition for speed
                loads[ccw_edges[i]] += demands[i][2]
            else:
                # CW
                routing[i] = 0.0
                loads[cw_edges[i]] += demands[i][2]
                
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_routing = routing.copy()
            
            # Early termination if we hit the theoretical lower bound?
            # Lower bound is sum(weights) / n_nodes = 35 / 10 = 3.5.
            # But max load is integer, so min possible is 4? No, loads are integers.
            # If best_max_load == 4, we can stop.
            # Actually, let's just let it run, 32768 is very fast.
            
    return best_routing
