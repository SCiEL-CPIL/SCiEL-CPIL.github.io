import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    # Rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x6 matrix.
    # M_{2,2,6} is the direct sum of three M_{2,2,2} tensors.
    # The rank of M_{2,2,2} is 7 (Strassen's algorithm is rank-optimal).
    # Therefore, the minimum rank for M_{2,2,6} is 3 * 7 = 21.
    # We construct the rank 21 decomposition directly using Strassen's 7-term algorithm
    # applied independently to each 2x2x2 block.
    
    n, m, p = DIMENSIONS
    rank = 21
    
    # Strassen decomposition for 2x2 matrices.
    # The 2x2x2 multiplication tensor T satisfies:
    #   sum_r U[r,i,j] * V[r,j,k] * W[r,i,k] = delta_{i,k} * delta_{j,k}  (bilinear form)
    # In flattened form (row-major 2x2):
    #   U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # where index 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    #
    # The 7 rank-1 terms of Strassen's algorithm:
    #   M1 = (A11 - A22) * (B11 + B22)
    #   M2 = (A11 + A22) * B22
    #   M3 = A12 * (B21 - B22)
    #   M4 = A21 * (B11 - B12)
    #   M5 = (A11 + A12) * B11
    #   M6 = (A21 - A11) * (B11 + B12)
    #   M7 = (A12 - A22) * (B21 + B22)
    #
    # U vectors (flattened 2x2, row-major: [00, 01, 10, 11]):
    #   u1 = [1, 0, 0, -1]
    #   u2 = [1, 0, 0, 1]
    #   u3 = [0, 1, 0, 0]
    #   u4 = [0, 0, 1, 0]
    #   u5 = [1, 1, 0, 0]
    #   u6 = [-1, 0, 1, 0]
    #   u7 = [0, 1, 0, -1]
    #
    # V vectors:
    #   v1 = [1, 0, 0, 1]
    #   v2 = [0, 0, 0, 1]
    #   v3 = [0, 0, 1, -1]
    #   v4 = [1, -1, 0, 0]
    #   v5 = [1, 0, 0, 0]
    #   v6 = [1, 1, 0, 0]
    #   v7 = [0, 0, 1, 1]
    #
    # W vectors:
    #   w1 = [1, 0, 0, -1]
    #   w2 = [1, 0, 0, 1]
    #   w3 = [0, 0, 0, 1]
    #   w4 = [0, 0, 1, 0]
    #   w5 = [1, 0, 1, 0]
    #   w6 = [0, 0, 1, 1]
    #   w7 = [0, 1, 0, 0]
    
    strassen_U = np.array([
        [1, 1, 0, 0, 1, -1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [-1, 1, 0, 0, 0, 0, -1]
    ], dtype=object)
    
    strassen_V = np.array([
        [1, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 1, -1, 0, 0, 0, 1]
    ], dtype=object)
    
    strassen_W = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0],
        [-1, 1, 1, 0, 0, 0, 0]
    ], dtype=object)
    
    # Initialize factor matrices
    # U: (n*m, rank) = (4, 21)
    # V: (m*p, rank) = (12, 21)
    # W: (p*n, rank) = (12, 21)
    # V index: j*p + k, where j in 0..1 (row of B), k in 0..5 (col of B)
    # W index: k*n + i, where k in 0..5 (row of C), i in 0..1 (col of C)
    
    U = np.zeros((4, rank), dtype=object)
    V = np.zeros((12, rank), dtype=object)
    W = np.zeros((12, rank), dtype=object)
    
    # The 2x2x6 multiplication consists of 3 independent 2x2x2 multiplications
    # corresponding to k-blocks: {0,1}, {2,3}, {4,5}
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            
            # Fill U: same pattern for all blocks
            for idx in range(4):
                U[idx, col] = strassen_U[idx, s]
            
            # Fill V: for block b, k in {2*b, 2*b+1}
            # V index for (j, k): j*p + k = j*6 + k
            v_indices = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                V[v_indices[idx], col] = strassen_V[idx, s]
            
            # Fill W: for block b, k in {2*b, 2*b+1}
            # W index for (k, i): k*n + i = k*2 + i
            w_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                W[w_indices[idx], col] = strassen_W[idx, s]
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
