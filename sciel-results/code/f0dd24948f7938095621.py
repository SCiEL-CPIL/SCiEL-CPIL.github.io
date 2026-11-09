"""Tensor decomposition for 4x5 matrix multiplication (rank 120, schoolbook)."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Vectorized Schoolbook decomposition using NumPy broadcasting.
    # This approach constructs the identity-like mappings for the tensor product
    # directly using array operations, distinct from the iterative loop approach.
    import numpy as np
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as zero matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Create indices for i, j, k
    i_vals = np.arange(n)
    j_vals = np.arange(m)
    k_vals = np.arange(p)
    
    # Generate all combinations of (i, j, k)
    # We need to map each (i, j, k) to a unique column index c
    # c = i * m * p + j * p + k
    # Row in U: i * m + j
    # Row in V: j * p + k
    # Row in W: k * n + i
    
    # Use meshgrid to get all combinations
    # To save memory, we can iterate over the outer dimension or use flat indices
    # Since n, m, p are small, we can create the full index arrays
    
    ii, jj, kk = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
    ii_flat = ii.flatten()
    jj_flat = jj.flatten()
    kk_flat = kk.flatten()
    
    # Calculate column indices
    col_indices = (ii_flat * m * p + jj_flat * p + kk_flat)
    
    # Calculate row indices for U, V, W
    u_rows = ii_flat * m + jj_flat
    v_rows = jj_flat * p + kk_flat
    w_rows = kk_flat * n + ii_flat
    
    # Set the 1s in the matrices
    # U[u_rows, col_indices] = 1
    # V[v_rows, col_indices] = 1
    # W[w_rows, col_indices] = 1
    
    U[u_rows, col_indices] = 1
    V[v_rows, col_indices] = 1
    W[w_rows, col_indices] = 1
    
    # Convert to lists to match the expected output format (list of lists)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
