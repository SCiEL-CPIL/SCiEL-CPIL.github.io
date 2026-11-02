import numpy as np

def construct():
    n, m, p = 2, 2, 16
    rank = n * m * p  # 64

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Generate all (i, j, k) combinations
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    
    # Flatten to get 1D arrays of indices
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Calculate column index c = (i*m + j)*p + k
    c_flat = (i_flat * m + j_flat) * p + k_flat
    
    # Calculate row indices for U, V, W
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Fill the matrices
    # U[u_rows, c_flat] = 1
    # V[v_rows, c_flat] = 1
    # W[w_rows, c_flat] = 1
    
    U[u_rows, c_flat] = 1
    V[v_rows, c_flat] = 1
    W[w_rows, c_flat] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
