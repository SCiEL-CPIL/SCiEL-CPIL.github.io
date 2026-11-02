"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145

    rank = n * m * p
    
    # Use vectorized index generation to create distinct internal logic
    # i, j, k ranges
    i_vec = np.repeat(np.arange(n), m * p)
    j_vec = np.tile(np.repeat(np.arange(m), p), n)
    k_vec = np.tile(np.arange(p), n * m)
    
    # Calculate row indices for U, V, W
    u_rows = i_vec * m + j_vec
    v_rows = j_vec * p + k_vec
    w_rows = k_vec * n + i_vec
    
    # Initialize dense integer matrices
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Set the 1s
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
