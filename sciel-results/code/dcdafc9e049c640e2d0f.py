import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Generate indices
    # Loop over k (13), j (2), i (2)
    # Total 52 columns
    
    # We can vectorize the assignment
    # Create arrays for all combinations of i, j, k
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Map to column index
    # col = k * (n * m) + j * n + i ?
    # The order in the loop was i, j, k.
    # Let's stick to the logic:
    # col = k * (n * m) + j * n + i
    # Wait, the previous code did:
    # for i, for j, for k: column += 1
    # So col = i * (m * p) + j * p + k
    # Let's check previous code:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       column += 1
    # Yes, col = i * (m * p) + j * p + k
    
    col_idx = i_flat * (m * p) + j_flat * p + k_flat
    
    # U index: i * m + j
    u_idx = i_flat * m + j_flat
    # V index: j * p + k
    v_idx = j_flat * p + k_flat
    # W index: k * n + i
    w_idx = k_flat * n + i_flat
    
    U[u_idx, col_idx] = 1
    V[v_idx, col_idx] = 1
    W[w_idx, col_idx] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
