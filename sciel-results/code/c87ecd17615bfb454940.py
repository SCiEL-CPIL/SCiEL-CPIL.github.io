# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem on a ring of 10 nodes with 10 edges,
    15 demands. Choose CW (0) or CCW (1) for each demand to minimize
    the maximum edge load.
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
    
    # Precompute for each demand: which edges are used if CW, which if CCW
    # CW from s to t: edges s, s+1, ..., t-1 (mod n)
    # CCW from s to t: all other edges
    
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        if s == t:
            cw_e = []
        else:
            cw_e = []
            e = s
            while e != t:
                cw_e.append(e)
                e = (e + 1) % n
        ccw_e = [e for e in range(n) if e not in cw_e]
        cw_edges.append(cw_e)
        ccw_edges.append(ccw_e)
    
    # Precompute weight contribution per edge for CW and CCW
    # cw_contrib[d][e] = weight if demand d is routed CW and uses edge e, else 0
    # ccw_contrib[d][e] = weight if demand d is routed CCW and uses edge e, else 0
    
    # For speed, precompute for each demand and each edge:
    # cw_w[d][e] = weight if edge e is in cw route of demand d, else 0
    # ccw_w[d][e] = weight if edge e is in ccw route of demand d, else 0
    
    cw_w = np.zeros((D, n))
    ccw_w = np.zeros((D, n))
    for d in range(D):
        s, t, w = demands[d]
        for e in cw_edges[d]:
            cw_w[d, e] = w
        for e in ccw_edges[d]:
            ccw_w[d, e] = w
    
    # Exhaustive search over 2^15 = 32768 configurations
    # For each configuration (bitmask), compute edge loads and max load
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 600  # 600s budget, use safety margin
    
    total_configs = 1 << D
    
    for mask in range(total_configs):
        if time.time() > deadline:
            break
        
        # Compute edge loads
        # For each demand d, if bit d of mask is 1, route CCW; else CW
        # Edge load[e] = sum over all demands d of (cw_w[d,e] if bit d=0 else ccw_w[d,e])
        
        # Efficient: load[e] = sum_d cw_w[d,e] + sum_d (ccw_w[d,e] - cw_w[d,e]) * bit_d
        # Let base[e] = sum_d cw_w[d,e]
        # Let diff[d,e] = ccw_w[d,e] - cw_w[d,e]
        # load[e] = base[e] + sum_d diff[d,e] * bit_d
        
        # Precompute base and diff
        # Actually, let's just compute directly for clarity and speed with numpy
        
        # Build the routing vector
        # routing[d] = 1 if bit d is set (CCW), 0 if not (CW)
        # load[e] = sum_d (routing[d] * ccw_w[d,e] + (1-routing[d]) * cw_w[d,e])
        #        = sum_d cw_w[d,e] + sum_d routing[d] * (ccw_w[d,e] - cw_w[d,e])
        
        # For this mask:
        routing = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.int8)
        # load = cw_w.sum(axis=0) + routing @ (ccw_w - cw_w)
        # But this creates arrays each iteration. Let's optimize.
        
        # Alternative: precompute base_load = cw_w.sum(axis=0) and diff = ccw_w - cw_w
        # Then load = base_load + routing @ diff
        
        # Let me precompute outside the loop
        pass
    
    # Better approach: precompute base and diff, then iterate
    base_load = cw_w.sum(axis=0)  # shape (n,)
    diff = ccw_w - cw_w  # shape (D, n)
    
    best_max_load = float('inf')
    best_mask = 0
    
    for mask in range(total_configs):
        if time.time() > deadline:
            break
        
        routing = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)
        load = base_load + routing @ diff
        max_load = load.max()
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_mask = mask
    
    # Extract best routing
    best_routing = np.array([(best_mask >> d) & 1 for d in range(D)], dtype=np.float64)
    
    return best_routing

# EVOLVE-BLOCK-END
