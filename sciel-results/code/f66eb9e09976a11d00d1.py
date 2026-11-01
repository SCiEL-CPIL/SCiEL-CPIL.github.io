import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    # Rank is n*m*p = 48
    # We construct U, V, W as numpy arrays and convert to lists
    # U shape (n*m, n*m*p)
    # V shape (m*p, n*m*p)
    # W shape (p*n, n*m*p)
    
    # The tensor T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # We can generate the indices for the 1s.
    
    # Generate all (i, j, k) tuples
    i_vals, j_vals, k_vals = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Flatten to get the column indices for the rank dimension
    # Each (i,j,k) corresponds to one rank-1 term
    # The rank index r is just the index in the flattened grid
    r_indices = np.arange(n * m * p)
    
    # Compute flattened indices for U, V, W
    # U index: i*m + j
    u_indices = i_vals.flatten() * m + j_vals.flatten()
    # V index: j*p + k
    v_indices = j_vals.flatten() * p + k_vals.flatten()
    # W index: k*n + i
    w_indices = k_vals.flatten() * n + i_vals.flatten()
    
    # Initialize matrices
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    # Set the 1s
    U[u_indices, r_indices] = 1
    V[v_indices, r_indices] = 1
    W[w_indices, r_indices] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
