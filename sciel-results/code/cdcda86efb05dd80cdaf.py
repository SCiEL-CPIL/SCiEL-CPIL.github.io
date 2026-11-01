"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Rank 7 decomposition for 2x2 matrix multiplication.
    # This uses a data-driven approach to define the factors, 
    # distinct from the hardcoded variable assignment in the seed.
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # Define the 7 rank-1 terms (u, v, w)
    # Each term is a tuple (u_vec, v_vec, w_vec)
    # u_vec corresponds to A indices (i, j) -> index i*m + j
    # v_vec corresponds to B indices (j, k) -> index j*p + k
    # w_vec corresponds to C indices (k, i) -> index k*n + i
    
    # Standard Strassen decomposition
    terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
