# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Uses exhaustive search over all 2^15 = 32768 possible routing assignments.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        0 -> clockwise, 1 -> counter-clockwise
    """
    n = 10
    D = 15
    
    # Demands (source, target, weight) in order
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
    
    # Precompute edge indices for clockwise and counter-clockwise routes
    cw_edges = []
    ccw_edges = []
    
    for (s, t, w) in demands:
        cw = []
        if s <= t:
            for e in range(s, t):
                cw.append(e)
        else:
            for e in range(s, n):
                cw.append(e)
            for e in range(0, t):
                cw.append(e)
        cw_edges.append(cw)
        
        all_edges = list(range(n))
        ccw = [e for e in all_edges if e not in cw]
        ccw_edges.append(ccw)
    
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    
    # Precompute contribution vectors
    cw_contrib = np.zeros((D, n), dtype=np.float64)
    ccw_contrib = np.zeros((D, n), dtype=np.float64)
    
    for d in range(D):
        w = weights[d]
        for e in cw_edges[d]:
            cw_contrib[d, e] = w
        for e in ccw_edges[d]:
            ccw_contrib[d, e] = w
    
    # Build a matrix where each row is a demand's contribution difference (CCW - CW)
    # When demand d is routed CCW, load change = ccw_contrib[d] - cw_contrib[d]
    # When demand d is routed CW, load change = 0
    # Total load = sum of cw_contrib for all demands + sum of (ccw_contrib[d] - cw_contrib[d]) for demands routed CCW
    
    base_loads = np.sum(cw_contrib, axis=0)  # All CW baseline
    delta = ccw_contrib - cw_contrib  # Change when switching from CW to CCW
    
    # Exhaustive search over all 2^15 combinations
    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.float64)
    
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0
    
    for mask in range(1 << D):
        if time.time() > deadline - safety_margin:
            break
        
        # Convert mask to routing vector
        routing = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)
        
        # Compute loads: base + sum of deltas for demands routed CCW
        ccw_mask = routing >= 0.5
        if np.any(ccw_mask):
            loads = base_loads + np.sum(delta[ccw_mask], axis=0)
        else:
            loads = base_loads.copy()
        
        max_load = np.max(loads)
        
        if max_load < best_max_load - 1e-12:
            best_max_load = max_load
            best_routing = routing.copy()
    
    return best_routing

# EVOLVE-BLOCK-END
