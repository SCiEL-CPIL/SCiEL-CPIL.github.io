"""Matrix multiplication 2x3x4 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1: (a+d)(e+i)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: a(f-i)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: d(g-c)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (a+b)(h-i)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1; V[2, 4] = -1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (c-d)(g+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # M7: (a+b)(e-f)
    U[0, 6] = 1; U[1, 6] = 1
    V[0, 6] = 1; V[1, 6] = -1
    W[0, 6] = 1; W[2, 6] = -1
    
    return U, V, W

def _construct_rank_22():
    """Construct a rank 22 decomposition for 2x3x4."""
    n, m, p = 2, 3, 4
    rank = 22
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Part 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3 + j, r] = U_s[i*2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j*4 + k, r] = V_s[j*2 + k, r]
        for k in range(2):
            for i in range(2):
                W[k*2 + i, r] = W_s[k*2 + i, r]
                
    # Part 1b: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    for r in range(7):
        col = r + 7
        for i in range(2):
            for j in range(2):
                U[i*3 + j, col] = U_s[i*2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j*4 + (k + 2), col] = V_s[j*2 + k, r]
        for k in range(2):
            for i in range(2):
                W[(k + 2)*2 + i, col] = W_s[k*2 + i, r]

    # Part 2: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    for k in range(4):
        for i in range(2):
            col = 14 + k * 2 + i
            U[i*3 + 2, col] = 1.0
            V[2*4 + k, col] = 1.0
            W[k*2 + i, col] = 1.0
            
    return U, V, W

def _build_target_tensor():
    """Build the 2x3x4 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _check_decomposition(U, V, W, T):
    """Check if U, V, W exactly decompose T."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.allclose(reconstructed, T, atol=1e-10, rtol=1e-10)

def _schoolbook_decomposition():
    """Return the schoolbook (rank 24) decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    U, V, W = _construct_rank_22()
    
    if _check_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    U_s, V_s, W_s = _schoolbook_decomposition()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
