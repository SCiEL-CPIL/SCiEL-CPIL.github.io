import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create index arrays for all combinations of i, j, k
    # i in [0, n), j in [0, m), k in [0, p)
    # We need to map these to the specific row indices for U, V, W
    
    # U has rows indexed by i*m + j
    # V has rows indexed by j*p + k
    # W has rows indexed by k*n + i
    
    # Generate all triples (i, j, k)
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Flatten the arrays to get the list of indices for each column in the decomposition
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Initialize U, V, W as zero matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Set the 1s in the appropriate positions
    # For each column r, U[u_rows[r], r] = 1, etc.
    U[u_rows, np.arange(rank)] = 1
    V[v_rows, np.arange(rank)] = 1
    W[w_rows, np.arange(rank)] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
