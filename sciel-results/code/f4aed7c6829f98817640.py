# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring loading problem for 10 nodes and 15 demands using exhaustive
    search over all 2^15 routing possibilities. Returns the optimal routing
    directions as a numpy array of shape (15,).
    
    0.0 indicates Clockwise routing.
    1.0 indicates Counter-Clockwise routing.
    """
    start_time = time.time()
    # Deadline guard: 600s limit, keep 10s margin
    deadline = start_time + 590.0
    
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
    # cw_loads[i] is a vector of length 10, where entry e is the weight if demand i uses edge e in CW direction
    cw_loads = np.zeros((D, n_nodes))
    ccw_loads = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise path: s, s+1, ..., t-1 (mod n)
        # Edges are indexed by the starting node.
        # If s < t, edges are s, ..., t-1
        # If s > t, edges are s, ..., n-1, 0, ..., t-1
        # If s == t, no edges (but demands are distinct nodes usually, here s!=t)
        
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

    # Generate all 2^D masks
    # Each mask is a vector of 0s and 1s of length D
    # 0 -> CW, 1 -> CCW
    # Use float32 to reduce memory usage and potentially speed up matmul
    masks = np.array([list(map(int, np.binary_repr(i, width=D))) for i in range(2**D)], dtype=np.float32)
    
    # Check time before heavy computation (though it's fast)
    if time.time() > deadline:
        # Fallback: return all clockwise (0.0) or a heuristic if time runs out
        # But 32k iterations is < 0.1s usually. This is just a safety net.
        return np.zeros(D, dtype=np.float32)

    # Compute total loads for all masks simultaneously
    loads_cw_part = (1.0 - masks) @ cw_loads
    loads_ccw_part = masks @ ccw_loads
    total_loads = loads_cw_part + loads_ccw_part
    
    # Find the mask with the minimum maximum load
    max_loads = np.max(total_loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    return best_mask

# EVOLVE-BLOCK-END
