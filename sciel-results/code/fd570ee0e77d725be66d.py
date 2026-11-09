# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solves the ring loading problem for 10 nodes and 15 demands.
    Uses exhaustive search to find the optimal routing that minimizes
    the maximum edge load.
    
    0.0 indicates Clockwise routing.
    1.0 indicates Counter-Clockwise routing.
    """
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
        (5, 2, 2)
    ]
    
    D = len(demands)
    
    # Pre-compute load vectors for each demand for CW and CCW routing
    cw_loads = np.zeros((D, n_nodes))
    ccw_loads = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        if s == t:
            continue
            
        if s < t:
            edges = range(s, t)
        else:
            edges = list(range(s, n_nodes)) + list(range(0, t))
            
        for e in edges:
            cw_loads[i, e] = w
            
        # Counter-Clockwise path: all other edges
        for e in range(n_nodes):
            if cw_loads[i, e] == 0:
                ccw_loads[i, e] = w

    # Exhaustive search over all 2^D possibilities
    # D=15, so 2^15 = 32768 possibilities, which is very fast
    best_max_load = np.inf
    best_mask = np.zeros(D)
    
    # Generate all masks
    # Using vectorized numpy operations for speed
    # Create binary representations for all 2^D cases
    masks = np.array([list(map(int, np.binary_repr(i, width=D))) for i in range(2**D)])
    
    # Calculate loads for all masks simultaneously
    # load = sum(mask[i] * ccw_loads[i] + (1-mask[i]) * cw_loads[i])
    # = sum(cw_loads[i]) + sum(mask[i] * (ccw_loads[i] - cw_loads[i]))
    base_load = np.sum(cw_loads, axis=0)
    diff_loads = ccw_loads - cw_loads
    
    # total_loads has shape (2^D, n_nodes)
    total_loads = base_load[None, :] + masks @ diff_loads
    
    # Find max load for each mask
    max_loads = np.max(total_loads, axis=1)
    
    # Find the mask with the minimum max load
    best_idx = np.argmin(max_loads)
    best_max_load = max_loads[best_idx]
    best_mask = masks[best_idx].astype(float)
    
    return best_mask

# EVOLVE-BLOCK-END
