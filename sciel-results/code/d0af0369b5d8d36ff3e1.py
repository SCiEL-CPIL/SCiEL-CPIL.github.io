# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem by exhaustive search (brute force) 
    over all 2^15 possible routing combinations using vectorized NumPy operations.
    """
    n_nodes = 10
    n_edges = 10
    D = 15

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

    # Precompute edge contribution matrices for CW and CCW routing
    # cw_contrib[d, e] = weight of demand d if routed CW, 0 otherwise
    # ccw_contrib[d, e] = weight of demand d if routed CCW, 0 otherwise
    cw_contrib = np.zeros((D, n_edges))
    ccw_contrib = np.zeros((D, n_edges))

    for d, (s, t, w) in enumerate(demands):
        cur = s
        while cur != t:
            cw_contrib[d, cur] = w
            cur = (cur + 1) % n_nodes
        
        # CCW is the complement of CW in the ring
        # Edges not in CW path are in CCW path
        mask_cw = (cw_contrib[d] > 0)
        mask_ccw = ~mask_cw
        ccw_contrib[d, mask_ccw] = w

    best_max_load = float('inf')
    best_routing = np.zeros(D)

    # Iterate over all 2^15 combinations
    # We can process multiple masks at a time or just vectorize the load calculation
    for mask in range(1 << D):
        # Determine routing decisions for all demands
        # bits = array of 0/1 for each demand, 1 means CCW
        bits = np.array([(mask >> d) & 1 for d in range(D)])
        
        # For each demand, if bit is 1, use ccw_contrib, else cw_contrib
        # We can use np.where
        # contrib = np.where(bits[:, None] == 1, ccw_contrib, cw_contrib)
        # loads = contrib.sum(axis=0)
        
        # More efficient:
        # loads = sum of cw_contrib for bits==0 + sum of ccw_contrib for bits==1
        # Or simply:
        # Select matrix rows based on bits
        # indices = np.where(bits == 1, 1, 0)
        # This is still a loop in Python for the mask iteration.
        
        # Let's try to vectorize the mask iteration itself?
        # 32768 masks is small. The inner vectorization is the key.
        
        bits = np.array([(mask >> d) & 1 for d in range(D)])
        # Create a selection matrix
        # If bit is 1, we want ccw_contrib[d], else cw_contrib[d]
        # We can do: contrib = cw_contrib * (1 - bits) + ccw_contrib * bits
        # This is element-wise multiplication and addition.
        contrib = cw_contrib * (1 - bits[:, None]) + ccw_contrib * bits[:, None]
        loads = contrib.sum(axis=0)
        
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_routing = bits.astype(float)

    return best_routing

# EVOLVE-BLOCK-END
