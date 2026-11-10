# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        < 0.5 -> route the demand clockwise
        >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n = 10
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
    D = len(demands)
    
    # Deadline guard: 630s limit, aim to finish well before.
    # This search is O(2^D * N) with vectorized ops, should be < 1s.
    deadline = time.time() + 590.0

    # Precompute masks
    # cw_mask[i, e] is True if edge e is used by demand i when routed CW
    # ccw_mask[i, e] is True if edge e is used by demand i when routed CCW
    
    def get_cw_edges(s, t):
        edges = np.zeros(n, dtype=bool)
        if s == t:
            edges[:] = True
            return edges
        e = s
        while e != t:
            edges[e] = True
            e = (e + 1) % n
        return edges

    cw_masks = np.zeros((D, n), dtype=bool)
    ccw_masks = np.zeros((D, n), dtype=bool)
    weights = np.zeros(D)
    
    for i, (s, t, w) in enumerate(demands):
        cw_masks[i] = get_cw_edges(s, t)
        ccw_masks[i] = ~cw_masks[i]
        weights[i] = w
        
    # Precompute delta vectors for switching direction
    # delta[i] = w_i * (ccw_mask_i - cw_mask_i)
    # If we switch from CW to CCW: loads += delta[i]
    # If we switch from CCW to CW: loads -= delta[i]
    # Note: ccw_mask - cw_mask is in {-1, 0, 1}
    # Since cw and ccw are complementary (partition of edges), 
    # ccw_mask = ~cw_mask.
    # So ccw_mask - cw_mask = (1 - cw_mask) - cw_mask = 1 - 2*cw_mask.
    # Let's just compute it directly to be safe.
    deltas = np.zeros((D, n), dtype=float)
    for i in range(D):
        deltas[i] = weights[i] * (ccw_masks[i].astype(float) - cw_masks[i].astype(float))
        
    # Initial state: all CW
    # loads = sum of w_i * cw_mask_i
    loads = np.zeros(n)
    for i in range(D):
        loads += weights[i] * cw_masks[i]
        
    best_max_load = np.max(loads)
    best_routing = np.zeros(D, dtype=np.int8)
    
    total_iterations = 1 << D
    
    # Gray code iteration
    # routing_bits tracks current direction (0=CW, 1=CCW)
    routing_bits = np.zeros(D, dtype=np.int8)
    
    for i in range(1, total_iterations):
        if time.time() > deadline:
            break
            
        bit_idx = (i & -i).bit_length() - 1
        
        # Flip bit
        old_dir = routing_bits[bit_idx]
        new_dir = 1 - old_dir
        routing_bits[bit_idx] = new_dir
        
        if new_dir == 1:
            # CW -> CCW
            loads += deltas[bit_idx]
        else:
            # CCW -> CW
            loads -= deltas[bit_idx]
            
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_routing = routing_bits.copy()
            
    # Construct final routing
    # routing[i] >= 0.5 means CCW (bit 1)
    # routing[i] < 0.5 means CW (bit 0)
    final_routing = best_routing.astype(float)
    
    return final_routing

# EVOLVE-BLOCK-END
