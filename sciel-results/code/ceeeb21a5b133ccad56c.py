import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Schoolbook decomposition via vectorized NumPy construction
    # U[i*m + j, c] corresponds to A[i, j]
    # V[j*p + k, c] corresponds to B[j, k]
    # W[k*n + i, c] corresponds to C[i, k] (task convention: W[k*n+i, r])
    
    # Generate all (i, j, k) triples
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    
    # Flatten to 1D arrays of indices
    i_flat = i_vals.ravel()
    j_flat = j_vals.ravel()
    k_flat = k_vals.ravel()
    
    # Compute row indices for each factor matrix
    u_rows = i_flat * m + j_flat  # shape (rank,)
    v_rows = j_flat * p + k_flat  # shape (rank,)
    w_rows = k_flat * n + i_flat  # shape (rank,)
    
    # Create sparse-like factor matrices using advanced indexing
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    cols = np.arange(rank)
    U[u_rows, cols] = 1.0
    V[v_rows, cols] = 1.0
    W[w_rows, cols] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
