"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Vectorized construction of the standard rank n*m*p decomposition.
    # This approach uses NumPy meshgrid to generate indices efficiently.
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate row indices for U, V, W based on the trace convention.
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Column indices correspond to the rank dimension.
    cols = np.arange(rank)
    
    # Initialize factor matrices.
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Populate the matrices with 1s at the calculated indices.
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Return as lists to match the seed program's output format.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
