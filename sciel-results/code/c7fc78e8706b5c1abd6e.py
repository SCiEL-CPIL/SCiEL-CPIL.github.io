# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges.
    Uses an exhaustive search over all 2^15 possible routing combinations to find
    the one that minimizes the maximum edge load.
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
    
    # Precompute edge usage vectors for CW and CCW
    cw_vecs = np.zeros((D, n_nodes), dtype=np.int8)
    ccw_vecs = np.zeros((D, n_nodes), dtype=np.int8)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        cur = s
        while cur != t:
            cw_vecs[i, cur] = 1
            cur = (cur + 1) % n_nodes
        ccw_vecs[i, :] = 1 - cw_vecs[i, :]
    
    # Precompute weighted vectors
    weighted_cw = weights[:, np.newaxis] * cw_vecs
    weighted_ccw = weights[:, np.newaxis] * ccw_vecs
    
    # base_load: load when all demands go CW
    base_load = np.sum(weighted_cw, axis=0)
    # diff_loads[i]: change in load when demand i switches from CW to CCW
    diff_loads = weighted_ccw - weighted_cw  # shape (D, n_nodes)
    
    # Generate all masks and extract routing bits vectorized
    masks = np.arange(1 << D, dtype=np.uint32)
    bit_positions = np.arange(D, dtype=np.uint32)
    # routing_matrix[j, i] = 1 if demand i is routed CCW in mask j
    routing_matrix = ((masks[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1).astype(np.float64)
    
    # Compute all loads at once via matrix multiply
    # all_loads[j] = base_load + routing_matrix[j] @ diff_loads
    all_loads = base_load[np.newaxis, :] + routing_matrix @ diff_loads
    
    # Find the mask with minimum maximum load
    max_loads = np.max(all_loads, axis=1)
    best_idx = np.argmin(max_loads)
    
    # Extract routing for best mask
    best_routing = routing_matrix[best_idx]
    
    return best_routing

# EVOLVE-BLOCK-END
