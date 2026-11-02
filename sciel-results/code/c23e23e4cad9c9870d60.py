"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for 2x2 matrix multiplication
    # U_s[i, r] corresponds to coefficient of A[i] in M_r
    # V_s[j, r] corresponds to coefficient of B[j] in M_r
    # W_s[k, r] corresponds to coefficient of M_r in C[k]
    
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V_s = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    for k in range(5):
        col_start = 7 * k
        col_end = col_start + 7
        
        # U is the same for all blocks since A is shared
        U[:, col_start:col_end] = U_s
        
        # V rows for block k
        # B is 2x10, flattened row-major: index = row * 10 + col
        # Block k uses cols 2k, 2k+1
        # B[0, 2k] -> 2k
        # B[0, 2k+1] -> 2k+1
        # B[1, 2k] -> 10 + 2k
        # B[1, 2k+1] -> 10 + 2k + 1
        rows_v = [2*k, 2*k+1, 10+2*k, 10+2*k+1]
        for idx, row_idx in enumerate(rows_v):
            V[row_idx, col_start:col_end] = V_s[idx]
            
        # W rows for block k
        # C is 10x2 in memory? No, C is 2x10.
        # Trace convention W[k*n+i, r] implies index = col * n + row
        # Block k uses cols 2k, 2k+1
        # C[0, 2k] -> 2k * 2 + 0 = 4k
        # C[0, 2k+1] -> (2k+1) * 2 + 0 = 4k+2
        # C[1, 2k] -> 2k * 2 + 1 = 4k+1
        # C[1, 2k+1] -> (2k+1) * 2 + 1 = 4k+3
        rows_w = [4*k, 4*k+2, 4*k+1, 4*k+3]
        for idx, row_idx in enumerate(rows_w):
            W[row_idx, col_start:col_end] = W_s[idx]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
