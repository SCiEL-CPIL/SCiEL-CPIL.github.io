import numpy as np
import time

def ring_routing() -> np.ndarray:
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
    
    # Precompute CW and CCW edge usage matrices
    cw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    ccw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
        # If s <= t: edges s, s+1, ..., t-1
        # If s > t: edges s, s+1, ..., n-1, 0, 1, ..., t-1
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            cw_matrix[i, e] = w
        
        # CCW uses all other edges
        all_edges = set(range(n_nodes))
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        for e in ccw_edges:
            ccw_matrix[i, e] = w
    
    # Base load: all demands routed clockwise
    base_loads = cw_matrix.sum(axis=0)
    
    # Delta matrix: change in load if demand i is routed CCW instead of CW
    delta_matrix = ccw_matrix - cw_matrix
    
    # Exhaustive search over all 2^D routing combinations
    best_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit
    
    total_masks = 2 ** D
    batch_size = 4096
    
    for batch_start in range(0, total_masks, batch_size):
        if time.time() > deadline:
            break
        
        batch_end = min(batch_start + batch_size, total_masks)
        masks = np.arange(batch_start, batch_end, dtype=np.int64)
        
        # Decode masks to routing vectors: r[i] = 1 if demand i is CCW (bit i set)
        bit_positions = np.arange(D, dtype=np.int64)
        r_batch = ((masks[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1).astype(np.float64)
        
        # Compute loads: base_loads + sum of deltas for CCW-routed demands
        loads_batch = r_batch @ delta_matrix + base_loads[np.newaxis, :]
        
        # Find max load for each routing combination
        max_loads = np.max(loads_batch, axis=1)
        
        # Track best
        local_best_idx = np.argmin(max_loads)
        local_best_load = max_loads[local_best_idx]
        
        if local_best_load < best_load:
            best_load = local_best_load
            best_routing = r_batch[local_best_idx].copy()
    
    return best_routing
