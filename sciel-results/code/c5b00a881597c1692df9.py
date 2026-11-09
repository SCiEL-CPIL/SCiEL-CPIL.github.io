import numpy as np

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
    
    cw_matrix = np.zeros((D, n_nodes))
    ccw_matrix = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            cw_matrix[i, e] = w
            
        all_edges = set(range(n_nodes))
        cw_set = set(cw_edges)
        ccw_list = [e for e in range(n_nodes) if e not in cw_set]
        for e in ccw_list:
            ccw_matrix[i, e] = w
            
    base_loads = cw_matrix.sum(axis=0)
    delta_matrix = ccw_matrix - cw_matrix
    
    # Vectorized brute force
    total_bits = 2 ** D
    all_bits = np.arange(total_bits, dtype=np.int32)
    bit_positions = np.arange(D)
    
    # Create routing matrix (total_bits x D)
    # Shape: (32768, 15)
    r_all = ((all_bits[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1).astype(np.float64)
    
    # Calculate loads for all combinations
    # Shape: (32768, 10)
    loads_all = r_all @ delta_matrix + base_loads[np.newaxis, :]
    
    # Find best combination
    max_loads_all = np.max(loads_all, axis=1)
    best_idx = np.argmin(max_loads_all)
    best_routing = r_all[best_idx].copy()
    
    return best_routing
