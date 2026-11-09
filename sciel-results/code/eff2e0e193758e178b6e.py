import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search over all 2^15 routing 
    combinations to find the one that minimizes the maximum edge load.
    """
    # Demand definitions: (source, target, weight)
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
    
    # Pre-compute load vectors for clockwise and counter-clockwise routing
    cw_loads = np.zeros((n_demands, n_nodes))
    ccw_loads = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise: edges s, s+1, ..., t-1 (mod n_nodes)
        current = s
        while current != t:
            cw_loads[i, current] += w
            current = (current + 1) % n_nodes
        
        # Counter-clockwise: all other edges
        # Sum of all edges is 0..9. CCW is the complement.
        # Total ring length is 10.
        # We can just compute the complement.
        ccw_loads[i, :] = cw_loads[i, :] # Start with CW
        # Actually, it's easier to just fill the other edges.
        ccw_loads[i, :] = 0
        for e in range(n_nodes):
            if cw_loads[i, e] == 0:
                ccw_loads[i, e] = w
    
    start_time = time.time()
    deadline = start_time + 590.0

    # Vectorized Exhaustive Search
    # Base load if all are CW
    base_load = np.sum(cw_loads, axis=0)
        
    # Delta load if we switch from CW to CCW for a specific demand
    # load(mask) = base_load + sum(mask_i * (ccw_loads[i] - cw_loads[i]))
    delta_loads = ccw_loads - cw_loads
        
    # Generate all 2^15 masks as a binary matrix (32768 x 15)
    # Column i corresponds to bit i of the mask, which controls demand i
    shifts = np.arange(n_demands)
    masks = (np.arange(1 << n_demands)[:, None] >> shifts) & 1
        
    # Compute loads for all masks: (32768 x 15) @ (15 x 10) -> (32768 x 10)
    all_loads = masks @ delta_loads + base_load
        
    # Find the mask with the minimum maximum load
    max_loads = np.max(all_loads, axis=1)
    best_idx = np.argmin(max_loads)
        
    # Check time guard
    if time.time() > deadline:
        # Fallback to all clockwise if timeout (should not happen)
        return np.zeros(n_demands)

    # Construct the routing array for the best mask
    # Bit i is 1 -> CCW (1.0), Bit i is 0 -> CW (0.0)
    best_mask = (best_idx >> shifts) & 1
    best_routing = best_mask.astype(float)
        
    return best_routing
