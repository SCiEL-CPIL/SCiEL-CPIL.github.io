"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Vectorized construction of the standard schoolbook decomposition.
    # Rank is n*m*p.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate all (i, j, k) indices
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Calculate column index for each term in the decomposition
    # c = i * m * p + j * p + k
    cols = (i_idx * m * p + j_idx * p + k_idx).flatten()
    
    # Calculate row indices for U, V, W
    u_rows = (i_idx * m + j_idx).flatten()
    v_rows = (j_idx * p + k_idx).flatten()
    w_rows = (k_idx * n + i_idx).flatten()
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set the 1s
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Return as lists to match the seed's return type
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
