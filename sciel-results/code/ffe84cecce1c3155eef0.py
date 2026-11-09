import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by minimizing the maximum edge load.
    Uses a vectorized exhaustive search over all 2^15 routing combinations.
    """
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute load contributions
    # load_contrib[i, 0, :] is load vector if demand i is CW
    # load_contrib[i, 1, :] is load vector if demand i is CCW
    load_contrib = np.zeros((n_demands, 2, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        cw_edges = []
        current = s
        while current != t:
            cw_edges.append(current)
            current = (current + 1) % n_nodes
        
        load_contrib[i, 0, cw_edges] = w
        
        # CCW edges are the complement of CW edges
        all_edges = np.arange(n_nodes)
        ccw_mask = np.ones(n_nodes, dtype=bool)
        ccw_mask[cw_edges] = False
        load_contrib[i, 1, ccw_mask] = w

    # Base load: all demands routed CW
    base_load = np.sum(load_contrib[:, 0, :], axis=0)
    
    # Delta: difference between CCW and CW load for each demand
    delta = load_contrib[:, 1, :] - load_contrib[:, 0, :] # shape (15, 10)
    
    # Generate all 2^15 masks
    masks = np.arange(1 << n_demands, dtype=np.uint32)
    
    # Extract bits: shape (32768, 15)
    # masks[:, None] has shape (32768, 1)
    # np.arange(n_demands)[None, :] has shape (1, 15)
    bits = (masks[:, None] >> np.arange(n_demands)[None, :]) & 1
    
    # Compute loads for all masks: shape (32768, 10)
    # loads = base_load + bits @ delta
    loads = base_load[None, :] + (bits @ delta)
    
    # Find the mask with the minimum maximum load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    # Construct result array
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0

    return result
