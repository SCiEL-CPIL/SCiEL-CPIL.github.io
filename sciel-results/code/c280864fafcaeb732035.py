import numpy as np

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # contrib_cw[d, e] is weight of demand d on edge e if routed CW
    # contrib_ccw[d, e] is weight of demand d on edge e if routed CCW
    contrib_cw = np.zeros((D, n), dtype=np.float64)
    contrib_ccw = np.zeros((D, n), dtype=np.float64)
    
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            contrib_cw[d, e] += w
            e = (e + 1) % n
            
        # CCW is the rest of the edges
        # Total weight is w.
        # Edges not in CW are in CCW.
        # We can just set contrib_ccw for edges not in CW.
        # Or simpler:
        # contrib_ccw[d, :] = w - contrib_cw[d, :]  # Wait, this assumes w is split? No.
        # The weight w is added to EACH edge on the path.
        # So if edge e is on CW path, load += w.
        # If edge e is on CCW path, load += w.
        # My current contrib_cw[d, e] is either 0 or w.
        # So contrib_ccw[d, e] should be w if e is NOT in CW path, else 0.
        
        # Let's re-calculate CCW properly
        cw_edges = set()
        e = s
        while e != t:
            cw_edges.add(e)
            e = (e + 1) % n
            
        for e in range(n):
            if e in cw_edges:
                contrib_ccw[d, e] = 0.0
            else:
                contrib_ccw[d, e] = w
                
    # Base loads if all CW
    # loads = sum(contrib_cw[d, :] for d)
    base_loads = np.sum(contrib_cw, axis=0)
    
    # Diff matrix: diff[e, d] = contrib_ccw[d, e] - contrib_cw[d, e]
    # loads = base_loads + diff @ x
    # where x is vector of 0s and 1s
    diff = (contrib_ccw - contrib_cw).T  # shape (n, D)
    
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)
    
    # Iterate all combinations
    for mask in range(1 << D):
        # Create x vector
        x = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)
        loads = base_loads + diff @ x
        ml = np.max(loads)
        if ml < best_max_load:
            best_max_load = ml
            best_routing = x.copy()
            
    return best_routing
