import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for multiplying
    a 3 by 4 matrix by a 4 by 7 matrix.
    
    Uses NumPy for vectorized construction of the schoolbook decomposition.
    Rank is n*m*p = 84.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create arrays
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Generate indices for all (i, j, k) combinations
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Flatten the indices to get column indices and row indices
    # We need to map (i, j, k) to a single column index col
    # And map to row indices for U, V, W
    cols = np.arange(rank)
    
    # Flatten the multi-indices
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Calculate row indices
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Assign 1s
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
