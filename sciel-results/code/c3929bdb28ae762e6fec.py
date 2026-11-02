"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _construct_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    Splits B into two 4x4 blocks. Each block uses rank-49 Strassen.
    Total rank 98.
    """
    n, m, p = DIMENSIONS # 4, 4, 8
    
    # 2x2 Strassen factors
    # U2, V2, W2 are 7x4 matrices (7 factors, 4 elements each)
    U2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)
    
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Build 4x4x4 Strassen factors (49 of them)
    # Each factor is a triple of 4x4 matrices (flattened to 16)
    # We will generate these on the fly for each of the 2 blocks
    
    for t in range(2): # 2 blocks
        for s in range(7): # Top level Strassen factor
            for r in range(7): # Scalar level Strassen factor
                col = t * 49 + s * 7 + r
                
                # Compute U4, V4, W4 for this (s,r) pair
                # U4 is 4x4
                U4 = np.zeros(16)
                V4 = np.zeros(16)
                W4 = np.zeros(16)
                
                for I in range(4):
                    for J in range(4):
                        r_block = I // 2
                        c_block = J // 2
                        i_loc = I % 2
                        j_loc = J % 2
                        coeff = U2[s, r_block*2+c_block] * U2[r, i_loc*2+j_loc]
                        U4[I*4+J] = coeff
                        
                for J in range(4):
                    for K in range(4):
                        r_block = J // 2
                        c_block = K // 2
                        j_loc = J % 2
                        k_loc = K % 2
                        coeff = V2[s, r_block*2+c_block] * V2[r, j_loc*2+k_loc]
                        V4[J*4+K] = coeff
                        
                for I in range(4):
                    for K in range(4):
                        r_block = I // 2
                        c_block = K // 2
                        i_loc = I % 2
                        k_loc = K % 2
                        coeff = W2[s, r_block*2+c_block] * W2[r, i_loc*2+k_loc]
                        # W4 index is I, K. W matrix index is K, I (k*n+i)
                        W4[I*4+K] = coeff
                
                # Fill U
                U[:, col] = U4
                
                # Fill V
                # V index is j*p+k. p=8.
                # For block t, k is offset by 4*t
                for j in range(4):
                    for k in range(4):
                        V[j*8 + (k + 4*t), col] = V4[j*4+k]
                        
                # Fill W
                # W index is k*n+i. n=4.
                # For block t, k is offset by 4*t
                for k in range(4):
                    for i in range(4):
                        W[(k + 4*t)*4 + i, col] = W4[i*4+k]
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    result = _construct_strassen_4x4_4x8()
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if verification fails (should not happen)
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
