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
        # Precompute the contribution of each demand to each edge for both CW and CCW routes.
        cw_contrib = np.zeros((n_demands, n_nodes))
        ccw_contrib = np.zeros((n_demands, n_nodes))
        
        for i, (s, t, w) in enumerate(demands):
            # Determine CW edges
            if s < t:
                edges = list(range(s, t))
            else:
                edges = list(range(s, n_nodes)) + list(range(0, t))
            
            for e in edges:
                cw_contrib[i, e] = w
                
            # Determine CCW edges (complement of CW edges)
            # CCW edges are all edges NOT in CW edges
            # We can just set ccw_contrib to w for all edges, then subtract cw_contrib?
            # No, CCW path is the other way around.
            # Total edges = 10. CW edges count = dist(s,t). CCW edges count = 10 - dist(s,t).
            # It is safer to explicitly construct or use the complement.
            cw_mask = np.zeros(n_nodes, dtype=bool)
            for e in edges:
                cw_mask[e] = True
            
            # CCW edges are where cw_mask is False
            ccw_mask = ~cw_mask
            ccw_contrib[i, ccw_mask] = w

        # Base load if all demands were routed CW
        base_load = np.sum(cw_contrib, axis=0)
        
        # Delta load for each demand if we switch from CW to CCW
        # load = base + sum(deltas[i] for i in CCW_set)
        # delta[i] = ccw_contrib[i] - cw_contrib[i]
        deltas = ccw_contrib - cw_contrib
        
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
