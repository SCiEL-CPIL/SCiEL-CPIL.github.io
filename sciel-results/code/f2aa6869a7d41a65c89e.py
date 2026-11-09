import numpy as np

def construct():
    n, m, p = 2, 3, 3
    rank = n * m * p
    
    # Generate all indices
    i_vals = np.arange(n)
    j_vals = np.arange(m)
    k_vals = np.arange(p)
    
    # Meshgrid to get all combinations
    # We want shape (n, m, p)
    I, J, K = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
    
    # Flatten to 1D arrays of length rank
    I_flat = I.flatten()
    J_flat = J.flatten()
    K_flat = K.flatten()
    
    # Compute linear indices for U, V, W
    # U index: i * m + j
    u_idx = I_flat * m + J_flat
    # V index: j * p + k
    v_idx = J_flat * p + K_flat
    # W index: k * n + i
    w_idx = K_flat * n + I_flat
    
    # The rank index r is just 0 to rank-1
    r_idx = np.arange(rank)
    
    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill matrices
    U[u_idx, r_idx] = 1.0
    V[v_idx, r_idx] = 1.0
    W[w_idx, r_idx] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
