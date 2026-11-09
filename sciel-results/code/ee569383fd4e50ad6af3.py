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
        >= 0.5 -> route the demand counter-clockwise
        < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
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
    
    # Precompute edge indices for each demand in CW and CCW directions
    cw_edge_indices = []
    ccw_edge_indices = []
    
    for s, t, w in demands:
        # Clockwise edges: s, s+1, ..., t-1 (mod n)
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        
        cw_edge_indices.append(cw_edges)
        ccw_edge_indices.append(ccw_edges)

    # Precompute load vectors for each demand in each direction
    cw_vecs = []
    ccw_vecs = []
    
    for i, (s, t, w) in enumerate(demands):
        cw_vec = np.zeros(n_nodes)
        ccw_vec = np.zeros(n_nodes)
        
        for e in cw_edge_indices[i]:
            cw_vec[e] = w
        for e in ccw_edge_indices[i]:
            ccw_vec[e] = w
            
        cw_vecs.append(cw_vec)
        ccw_vecs.append(ccw_vec)
        
    # Base load: all demands routed CW
    base_loads = np.sum(cw_vecs, axis=0)
    
    # Delta load: change in load when switching demand i from CW to CCW
    # delta[i] = ccw_vec[i] - cw_vec[i]
    deltas = [ccw_vecs[i] - cw_vecs[i] for i in range(D)]
    
    best_load = float('inf')
    best_routing_mask = 0  # Bitmask: 1 for CCW, 0 for CW
    
    # Use Gray Code to iterate through all 2^D combinations efficiently
    current_loads = base_loads.copy()
    
    # Initial state: all 0s (all CW)
    if np.max(current_loads) < best_load:
        best_load = np.max(current_loads)
        best_routing_mask = 0
        
    # Deadline guard: 600s limit, we use 590s to be safe
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Iterate through Gray code sequence
    for i in range(1, 2**D):
        if time.time() > deadline:
            break
            
        # Find the bit that flips: index of least significant 1 in i
        changed_bit = (i & -i).bit_length() - 1
        
        # Determine if the bit turned 1 (CW->CCW) or 0 (CCW->CW)
        # Check the bit in the new Gray code value
        new_gray = i ^ (i >> 1)
        
        if (new_gray >> changed_bit) & 1:
            # Turned ON (CW -> CCW): Add delta
            current_loads += deltas[changed_bit]
        else:
            # Turned OFF (CCW -> CW): Subtract delta
            current_loads -= deltas[changed_bit]
            
        max_load = np.max(current_loads)
        if max_load < best_load:
            best_load = max_load
            best_routing_mask = new_gray
            
    # Convert best_routing_mask to array
    best_routing = np.zeros(D)
    for i in range(D):
        if (best_routing_mask >> i) & 1:
            best_routing[i] = 1.0
        else:
            best_routing[i] = 0.0
            
    return best_routing

# EVOLVE-BLOCK-END
