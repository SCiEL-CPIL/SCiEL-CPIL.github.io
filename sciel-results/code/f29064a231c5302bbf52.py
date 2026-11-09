import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by vectorized exhaustive search over all 
    2^15 routing combinations to find the one that minimizes the maximum edge load.
    """
    start_time = time.time()
    # 600s limit, keep a safe margin of 10s
    deadline = start_time + 590.0 

    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute contributions
    # cw_contrib[i, e] is weight of demand i if routed CW, 0 otherwise
    # ccw_contrib[i, e] is weight of demand i if routed CCW, 0 otherwise
    cw_contrib = np.zeros((n_demands, n_nodes), dtype=np.int32)
    ccw_contrib = np.zeros((n_demands, n_nodes), dtype=np.int32)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_contrib[i, current] = w
            current = (current + 1) % n_nodes
        
        # CCW uses all edges not in CW path
        # Since weights are positive, and edges are disjoint in terms of usage for a single demand
        # cw_contrib[i, e] is either 0 or w.
        # ccw_contrib[i, e] is w if cw_contrib[i, e] is 0, else 0.
        ccw_contrib[i, :] = w
        ccw_contrib[i, cw_contrib[i] > 0] = 0
        
    # Base load: all demands routed CW
    base_loads = np.sum(cw_contrib, axis=0)
    
    # Delta load: change in load for edge e if demand i switches from CW to CCW
    # delta[i, e] = ccw_contrib[i, e] - cw_contrib[i, e]
    delta_loads = ccw_contrib - cw_contrib
    
    # Vectorized exhaustive search
    # Generate all 2^15 masks
    n_masks = 1 << n_demands
    masks = np.arange(n_masks, dtype=np.uint16)
    
    # Generate direction matrix (n_masks, n_demands)
    # Bit i of mask determines direction for demand i
    # 1 if bit is set (CCW), 0 otherwise (CW)
    # Using broadcasting for speed
    directions = ((masks[:, None] >> np.arange(n_demands, dtype=np.uint16)) & 1).astype(np.int32)
    
    # Calculate loads for all masks
    # loads = base_loads + directions @ delta_loads
    # directions: (n_masks, n_demands)
    # delta_loads: (n_demands, n_nodes)
    # result: (n_masks, n_nodes)
    loads = base_loads[None, :] + directions @ delta_loads
    
    # Find mask with minimum max load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_routing = directions[best_idx].astype(np.float64)
    
    # Check deadline (though this should finish in milliseconds)
    if time.time() > deadline:
        # Fallback to a simple heuristic if somehow we are slow (should not happen)
        return np.zeros(n_demands, dtype=np.float64)
        
    return best_routing
