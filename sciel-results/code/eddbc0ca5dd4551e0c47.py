import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem using a vectorized exhaustive search.
    This approach computes the edge loads for all 2^15 routing combinations
    simultaneously using matrix operations, ensuring optimal performance and
    correctness for this problem size.
    """
    # Demand definitions: (source, target, weight)
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
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute CW and CCW edge usage matrices
    # cw_matrix[i, e] = 1 if edge e is used by demand i when routed CW, else 0
    # ccw_matrix[i, e] = 1 if edge e is used by demand i when routed CCW, else 0
    cw_matrix = np.zeros((n_demands, n_nodes), dtype=np.int8)
    ccw_matrix = np.zeros((n_demands, n_nodes), dtype=np.int8)
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1
        current = s
        while current != t:
            cw_matrix[i, current] = 1
            current = (current + 1) % n_nodes
        
        # Counter-clockwise edges: complement of CW edges
        # Since it's a ring, if an edge is not in CW path, it is in CCW path
        # Note: The union of CW and CCW edges is all 10 edges.
        ccw_matrix[i, :] = 1 - cw_matrix[i, :]
        
    weights = np.array([w for s, t, w in demands], dtype=np.float64)
    
    # Base load if all demands are routed CW
    # Base[e] = sum_i weights[i] * cw_matrix[i, e]
    base_loads = cw_matrix.T @ weights
    
    # Difference matrix W[i, e] = weights[i] * (ccw_matrix[i, e] - cw_matrix[i, e])
    # If we switch demand i from CW to CCW, the load change on edge e is W[i, e]
    diff_matrix = weights[:, np.newaxis] * (ccw_matrix - cw_matrix)
    
    # Generate all masks
    # We need a matrix A of shape (2^15, 15) where A[m, i] is 1 if bit i of m is set
    # This can be generated efficiently
    masks = np.arange(1 << n_demands, dtype=np.int32)
    
    # Create the bit matrix A
    # A[m, i] = (masks[m] >> i) & 1
    # Using broadcasting:
    # powers = 1 << np.arange(n_demands)
    # A = (masks[:, np.newaxis] & powers[np.newaxis, :]) != 0
    powers = 1 << np.arange(n_demands, dtype=np.int32)
    bit_matrix = (masks[:, np.newaxis] & powers[np.newaxis, :]) != 0
    
    # Compute the total load for all masks
    # TotalLoad[m, e] = base_loads[e] + sum_i bit_matrix[m, i] * diff_matrix[i, e]
    # This is base_loads + (bit_matrix @ diff_matrix)
    
    # Check deadline
    start_time = time.time()
    deadline = start_time + 600.0
    
    if time.time() > deadline:
        # Should not happen, but safety
        return np.zeros(n_demands)
        
    # Perform matrix multiplication
    # bit_matrix is (32768, 15), diff_matrix is (15, 10)
    # Result is (32768, 10)
    load_changes = bit_matrix @ diff_matrix
    total_loads = base_loads[np.newaxis, :] + load_changes
    
    # Find the mask with the minimum maximum edge load
    max_loads = np.max(total_loads, axis=1)
    min_max_load_idx = np.argmin(max_loads)
    
    best_mask = masks[min_max_load_idx]
    
    # Construct the routing array
    # Entry >= 0.5 means CCW (bit 1), < 0.5 means CW (bit 0)
    best_routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            best_routing[i] = 1.0
        else:
            best_routing[i] = 0.0
            
    return best_routing
