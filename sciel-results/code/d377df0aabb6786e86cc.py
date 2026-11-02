"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-98 decomposition for 4x4 by 4x8 matrix multiplication.
    Uses recursive Strassen via Kronecker product of 2x2x2 rank-7 decomposition.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Strassen 2x2x2 decomposition: 7 rank-1 terms
    # Indexing: row-major flattening of 2x2: [00, 01, 10, 11]
    
    U2 = np.array([
        [1, 0, 0, 1],   # A00+A11
        [0, 0, 1, 1],   # A10+A11
        [1, 0, 0, 0],   # A00
        [0, 0, 0, 1],   # A11
        [1, 1, 0, 0],   # A00+A01
        [-1, 0, 1, 0],  # A10-A00
        [0, 1, 0, -1]   # A01-A11
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1],   # B00+B11
        [1, 0, 0, 0],   # B00
        [0, 1, 0, -1],  # B01-B11
        [-1, 0, 1, 0],  # B10-B00
        [0, 0, 0, 1],   # B11
        [1, 1, 0, 0],   # B00+B01
        [0, 0, 1, 1]    # B10+B11
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1],   # C00+C11
        [0, 0, 1, -1],  # C10-C11
        [0, 1, 0, 1],   # C01+C11
        [1, 0, 1, 0],   # C00+C10
        [-1, 1, 0, 0],  # -C00+C01
        [0, 0, 0, 1],   # C11
        [1, 0, 0, 0]    # C00
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    def idx2(i, j):
        return i * 2 + j
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U4: 4x4 matrix entries for input A
            for I in range(4):
                for J in range(4):
                    r, c = I // 2, J // 2
                    i_loc, j_loc = I % 2, J % 2
                    coeff = U2[s, idx2(r, c)] * U2[t, idx2(i_loc, j_loc)]
                    U4[I * 4 + J, col] = coeff
            
            # V4: 4x4 matrix entries for input B
            for J in range(4):
                for K in range(4):
                    r, c = J // 2, K // 2
                    j_loc, k_loc = J % 2, K % 2
                    coeff = V2[s, idx2(r, c)] * V2[t, idx2(j_loc, k_loc)]
                    V4[J * 4 + K, col] = coeff
            
            # W4: 4x4 matrix entries for output C[I,K]
            for I in range(4):
                for K in range(4):
                    r, c = I // 2, K // 2
                    i_loc, k_loc = I % 2, K % 2
                    coeff = W2[s, idx2(r, c)] * W2[t, idx2(i_loc, k_loc)]
                    W4[I * 4 + K, col] = coeff
    
    # Extend to 4x4x8: two copies of the 4x4x4 decomposition
    rank = 98
    
    # U: shape (n*m, rank) = (16, 98)
    # V: shape (m*p, rank) = (32, 98)  
    # W: shape (p*n, rank) = (32, 98)
    # W[k*n+i, r] convention: output index is k*n+i where k is output col, i is output row
    
    U = np.zeros((16, 98), dtype=float)
    V = np.zeros((32, 98), dtype=float)
    W = np.zeros((32, 98), dtype=float)
    
    # Left block (B columns 0-3, C columns 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[i * 4 + k, :]
    
    # Right block (B columns 4-7, C columns 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 49:98] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:98] = W4[i * 4 + k, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
