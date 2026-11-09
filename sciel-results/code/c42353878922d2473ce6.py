# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using vectorized exhaustive search.
    For 15 demands, there are 2^15 = 32768 possible routings.
    This is small enough to enumerate all possibilities using NumPy vectorization,
    which is faster and more robust than invoking an external ILP solver.
    """
    # Problem data
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)  # 15
    n_edges = n_nodes # 10 edges
    
    # Precompute CW masks for each demand
    # cw_mask[i] is a boolean array of length 10, True if edge e is in CW path of demand i
    cw_masks = np.zeros((D, n_edges), dtype=bool)
    
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            cw_masks[i, s:t] = True
        else:
            cw_masks[i, s:] = True
            cw_masks[i, :t] = True
            
    # Weights
    weights = np.array([w for _, _, w in demands])
    
    # Total combinations
    n_combos = 1 << D
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0 # 600s limit, we should be much faster
    
    # We will iterate in chunks to avoid huge memory spikes if needed, 
    # but 32768 * 10 * 8 bytes is ~2.6 MB, so we can do it all at once.
    
    # Generate all routing combinations as bits
    # x: (n_combos, D)
    # x[k, i] is 1 if demand i is routed CCW, 0 if CW
    
    # Using np.indices or bit manipulation
    # Let's use bit manipulation for speed and memory efficiency
    # We can generate the array of bits directly.
    
    # Actually, let's just loop over combinations if vectorizing the bit generation is complex,
    # but vectorizing the LOAD calculation is key.
    
    # Alternative:
    # For each edge e, the load is sum of weights of demands that use it.
    # A demand uses edge e if:
    # (routed CW and e in CW path) OR (routed CCW and e in CCW path)
    # CCW path is complement of CW path.
    
    # Load_e = sum_i w_i * ( x_i * (1 - cw_mask_ie) + (1 - x_i) * cw_mask_ie )
    # Load_e = sum_i w_i * ( cw_mask_ie + x_i * (1 - 2*cw_mask_ie) )
    
    # Let B_ie = w_i * (1 - 2*cw_mask_ie)
    # Let C_e = sum_i w_i * cw_mask_ie
    
    # Load_e(x) = C_e + sum_i B_ie * x_i
    
    # This is a linear function of x.
    # We want to minimize max_e Load_e(x).
    
    # Let's construct the matrix B (D x n_edges) and vector C (n_edges)
    B = np.zeros((D, n_edges))
    C = np.zeros(n_edges)
    
    for i in range(D):
        w = weights[i]
        mask = cw_masks[i]
        C += w * mask
        # B[i, e] = w * (1 - 2 * mask[e])
        B[i, :] = w * (1 - 2 * mask)
        
    # Now, for a given x (vector of 0/1), Loads = C + x @ B
    # We need to find x that minimizes max(Loads)
    
    # Since 2^15 is small, we can enumerate x.
    # To do this efficiently, we can use the fact that x @ B is a linear combination of rows of B.
    
    # Let's generate all x vectors.
    # x_k = k in binary.
    
    # We can compute Loads for all k at once?
    # Loads (n_combos, n_edges) = C + (X @ B)
    # X is (n_combos, D)
    
    # Creating X of size 32768 x 15 is fine.
    
    # Bit extraction
    # x[k, i] = (k >> i) & 1
    
    # We can create X using broadcasting
    # k_vals = np.arange(n_combos)
    # bits = np.arange(D)
    # X = (k_vals[:, None] >> bits[None, :]) & 1
    
    # This creates an array of size 32768 x 15.
    
    k_vals = np.arange(n_combos, dtype=np.int32)
    bits = np.arange(D, dtype=np.int32)
    X = (k_vals[:, None] >> bits[None, :]) & 1
    
    # Calculate Loads
    # Loads = C[None, :] + X @ B
    Loads = C[None, :] + X @ B
    
    # Find max load for each routing
    max_loads = np.max(Loads, axis=1)
    
    # Find index of minimum max load
    best_idx = np.argmin(max_loads)
    
    # Extract best x
    x_best = X[best_idx]
    
    # Convert to routing vector
    # x_i = 1 -> CCW (value >= 0.5)
    # x_i = 0 -> CW (value < 0.5)
    routing = x_best.astype(float)
    
    # Ensure it's a 1D array of floats
    return routing

# EVOLVE-BLOCK-END
