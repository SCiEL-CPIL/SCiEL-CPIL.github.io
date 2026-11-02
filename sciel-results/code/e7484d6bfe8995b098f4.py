import numpy as np

def construct():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients
    u_coeffs = np.array([
        [[1, 0], [0, 1]], 
        [[0, 0], [1, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ])
    
    v_coeffs = np.array([
        [[1, 0], [0, 1]], 
        [[1, 0], [0, 0]], 
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]], 
        [[1, 1], [0, 0]], 
        [[0, 0], [1, 1]]
    ])
    
    w_coeffs = np.array([
        [[1, 0], [0, 1]], 
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]], 
        [[1, 0], [1, 0]], 
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]], 
        [[1, 0], [0, 0]]
    ])

    num_blocks = 4
    
    # Vectorized filling
    # U: shape (4, 28). u_coeffs: (7, 2, 2).
    # We need to repeat u_coeffs for each block.
    # U[:, block*7 : block*7+7] = u_coeffs.T.reshape(4, 7) ?
    # u_coeffs[r, i, j]. We want U[i*m+j, block*7+r] = u_coeffs[r, i, j].
    # Let's reshape u_coeffs to (4, 7) where index (i,j) maps to row, r maps to col.
    # u_coeffs is (7, 2, 2).
    # u_flat = u_coeffs.transpose(1, 2, 0).reshape(4, 7)
    
    u_flat = u_coeffs.transpose(1, 2, 0).reshape(4, 7)
    for b in range(num_blocks):
        U[:, b*7 : b*7+7] = u_flat
        
    # V: shape (16, 28). v_coeffs: (7, 2, 2).
    # V[j*p + k, col] = v_coeffs[r, j, k_local]
    # k_global = b*2 + k_local.
    # For a specific block b, the columns are b*7 to b*7+7.
    # The rows involved are j=0,1 and k_global in [2b, 2b+1].
    # Rows: 2b, 2b+1, 2b+2, 2b+3? No.
    # V index is j*p + k. p=8.
    # j=0, k=2b -> index 2b.
    # j=0, k=2b+1 -> index 2b+1.
    # j=1, k=2b -> index 8+2b.
    # j=1, k=2b+1 -> index 8+2b+1.
    
    # v_flat for a block:
    # We want a 4x7 matrix where row 0 corresponds to (j=0, k_local=0), row 1 to (j=0, k_local=1), etc.
    # v_coeffs[r, j, k_local].
    # v_block_flat = v_coeffs.transpose(1, 2, 0).reshape(4, 7)
    
    v_flat = v_coeffs.transpose(1, 2, 0).reshape(4, 7)
    
    for b in range(num_blocks):
        # Rows: 2b, 2b+1, 8+2b, 8+2b+1
        rows = [2*b, 2*b+1, 8+2*b, 8+2*b+1]
        V[rows, b*7 : b*7+7] = v_flat
        
    # W: shape (16, 28). w_coeffs: (7, 2, 2).
    # W[k*n + i, col] = w_coeffs[r, i, k_local]
    # k_global = b*2 + k_local.
    # W index is k*n + i. n=2.
    # k_global=2b, i=0 -> 4b.
    # k_global=2b, i=1 -> 4b+1.
    # k_global=2b+1, i=0 -> 4b+2.
    # k_global=2b+1, i=1 -> 4b+3.
    
    w_flat = w_coeffs.transpose(1, 2, 0).reshape(4, 7)
    
    for b in range(num_blocks):
        rows = [4*b, 4*b+1, 4*b+2, 4*b+3]
        W[rows, b*7 : b*7+7] = w_flat

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
