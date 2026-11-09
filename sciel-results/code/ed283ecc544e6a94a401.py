import numpy as np
import time

def ring_routing() -> np.ndarray:
    start_time = time.time()
    # Safety margin: return at least 10 seconds before the 600s hard limit
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
        (5, 2, 2),
    ]
    
    D = len(demands)
    total_configs = 1 << D
    
    # Precompute edge indices and weights for CW and CCW
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        weights.append(w)
        cw = []
        if s == t:
            # If source equals target, the path is ambiguous in standard definitions,
            # but typically implies 0 load or full ring. Given the "s to t-1" rule,
            # it's an empty set. However, in ring routing problems, s!=t is standard.
            # If s==t, we assume 0 edges for CW and 0 edges for CCW? 
            # Or maybe the whole ring? 
            # The problem statement says "Clockwise from s to t uses edges s, ..., t-1".
            # If s==t, the range is empty. So CW load is 0. CCW is "all other edges" -> all 10 edges?
            # Let's check the data: no s==t cases. So this branch is likely not hit.
            pass
        else:
            edge = s
            while edge != t:
                cw.append(edge)
                edge = (edge + 1) % n_nodes
        
        cw_edges.append(cw)
        ccw = [e for e in range(n_nodes) if e not in set(cw)]
        ccw_edges.append(ccw)
        
    # Initialize loads for mask 0 (all CW)
    loads = np.zeros(n_nodes, dtype=np.int64)
    for i in range(D):
        for e in cw_edges[i]:
            loads[e] += weights[i]
            
    best_max_load = np.max(loads)
    best_mask = 0
    
    # Gray code enumeration
    # gray = step ^ (step >> 1)
    # We iterate step from 1 to total_configs - 1
    # For each step, we find the bit that changed from prev_gray to gray
    # and update loads accordingly.
    
    prev_gray = 0
    current_gray = 0
    
    for step in range(1, total_configs):
        # Check time
        if time.time() > deadline:
            break
            
        current_gray = step ^ (step >> 1)
        
        # Find the changed bit
        diff = current_gray ^ prev_gray
        # diff has exactly one bit set
        # Find the index of that bit
        # Using bit_length is efficient
        bit_idx = (diff & -diff).bit_length() - 1
        
        # Determine the new state of bit_idx in current_gray
        is_ccw = (current_gray >> bit_idx) & 1
        
        # Update loads
        # If is_ccw is 1, the demand flipped to CCW. It was CW before.
        # So we subtract CW contribution and add CCW contribution.
        # If is_ccw is 0, the demand flipped to CW. It was CCW before.
        # So we subtract CCW contribution and add CW contribution.
        
        w = weights[bit_idx]
        
        if is_ccw:
            # Remove CW
            for e in cw_edges[bit_idx]:
                loads[e] -= w
            # Add CCW
            for e in ccw_edges[bit_idx]:
                loads[e] += w
        else:
            # Remove CCW
            for e in ccw_edges[bit_idx]:
                loads[e] -= w
            # Add CW
            for e in cw_edges[bit_idx]:
                loads[e] += w
                
        prev_gray = current_gray
        
        current_max = np.max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = current_gray
            
    # Construct the final routing array
    # Entry >= 0.5 routes CCW, < 0.5 routes CW.
    # In our mask, bit 1 means CCW.
    routing = np.zeros(D, dtype=float)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
