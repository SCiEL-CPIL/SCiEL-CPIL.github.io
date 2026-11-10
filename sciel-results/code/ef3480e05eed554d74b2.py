import numpy as np
import time

def ring_routing() -> np.ndarray:
    # Demands: (source, target, weight)
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
    start_time = time.time()
    deadline = start_time + 590  # 590s deadline with safety margin for 600s limit
    
    # Fallback: all CW routing
    fallback = np.zeros(n_demands)
    
    try:
        # Precompute deltas: change in load if demand i switches from CW to CCW
        # delta[i, e] = w if e in CCW path, -w if e in CW path, 0 otherwise
        deltas = np.zeros((n_demands, n_nodes))
        base_load = np.zeros(n_nodes)
        
        for i, (s, t, w) in enumerate(demands):
            # Determine CW edges
            if s < t:
                cw_edges = list(range(s, t))
            else:
                cw_edges = list(range(s, n_nodes)) + list(range(0, t))
            
            # Add to base load (all CW initially)
            for e in cw_edges:
                base_load[e] += w
                
            # Set deltas
            # For CW edges, switching to CCW removes w (delta = -w)
            for e in cw_edges:
                deltas[i, e] = -w
                
            # For CCW edges, switching to CCW adds w (delta = +w)
            # CCW edges are all edges not in CW edges
            cw_set = set(cw_edges)
            for e in range(n_nodes):
                if e not in cw_set:
                    deltas[i, e] = w

        # Vectorized exhaustive search
        # 2^15 is small enough to do all at once
        masks = np.arange(1 << n_demands)
        bits = (masks[:, None] >> np.arange(n_demands)[None, :]) & 1
        
        # Calculate loads for all masks: shape (2^n_demands, n_nodes)
        loads = base_load + bits @ deltas
        
        # Find the mask with minimum max load
        max_loads = loads.max(axis=1)
        best_idx = max_loads.argmin()
        best_mask = masks[best_idx]
        
        # Construct result array
        # Entry >= 0.5 means CCW (1), < 0.5 means CW (0)
        routing = np.zeros(n_demands)
        for i in range(n_demands):
            if (best_mask >> i) & 1:
                routing[i] = 1.0
            else:
                routing[i] = 0.0
                
        # Check deadline
        if time.time() > deadline:
            return fallback
            
        return routing
        
    except Exception:
        return fallback
