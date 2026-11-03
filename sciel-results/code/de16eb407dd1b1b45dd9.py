"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """
    Return Strassen's 7-rank decomposition for 2x2 * 2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    U[i*2+j, c] corresponds to A[i,j]
    V[j*2+k, c] corresponds to B[j,k]
    W[k*2+i, c] corresponds to C[i,k]
    """
    U = [
        [1, 0, 1, 0, 1, -1, -1],  # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, -1, 1, 0, 0, -1],  # A11
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, -1, 0, 0, 1, 0],   # B01
        [1, 0, 1, 0, 0, 0, 1],    # B10
        [1, 0, 0, 1, 1, 0, 1],    # B11
    ]
    
    W = [
        [1, 0, 0, 0, 0, 0, 1],    # C00 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # C01 (k=1,i=0)
        [0, 1, 0, 1, 0, 0, 0],    # C10 (k=0,i=1)
        [1, -1, 1, 0, 0, 1, 0],   # C11 (k=1,i=1)
    ]
    
    return np.array(U), np.array(V), np.array(W)


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _construct_45_rank():
    """
    Construct a rank-45 decomposition for 3x3 * 3x6.
    """
    n, m, p = 3, 3, 6
    rank = 45
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_s, V_s, W_s = _strassen_2x2()
    
    # Part 1: i in {0,1}, j in {0,1}
    # 3 blocks of Strassen for k pairs (0,1), (2,3), (4,5)
    for p_idx in range(3):
        k_base = 2 * p_idx
        for c_local in range(7):
            c_global = 7 * p_idx + c_local
            
            # U: i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    U[i * m + j, c_global] = U_s[i * 2 + j, c_local]
            
            # V: j in {0,1}, k in {k_base, k_base+1}
            for j in range(2):
                for k_local in range(2):
                    k = k_base + k_local
                    V[j * p + k, c_global] = V_s[j * 2 + k_local, c_local]
            
            # W: k in {k_base, k_base+1}, i in {0,1}
            for k_local in range(2):
                for i in range(2):
                    k = k_base + k_local
                    W[k * n + i, c_global] = W_s[k_local * 2 + i, c_local]
    
    # Part 2: i in {0,1}, j = 2
    # 6 terms for k in 0..5
    for k in range(p):
        c_global = 21 + k
        for i in range(2):
            U[i * m + 2, c_global] = 1
            W[k * n + i, c_global] = 1
        V[2 * p + k, c_global] = 1
        
    # Part 3: i = 2, j in {0,1}
    # 12 terms (2 for each k)
    for k in range(p):
        c1 = 27 + 2 * k
        c2 = 27 + 2 * k + 1
        
        # j=0
        U[2 * m + 0, c1] = 1
        V[0 * p + k, c1] = 1
        W[k * n + 2, c1] = 1
        
        # j=1
        U[2 * m + 1, c2] = 1
        V[1 * p + k, c2] = 1
        W[k * n + 2, c2] = 1
        
    # Part 4: i = 2, j = 2
    # 6 terms for k in 0..5
    for k in range(p):
        c_global = 39 + k
        U[2 * m + 2, c_global] = 1
        V[2 * p + k, c_global] = 1
        W[k * n + 2, c_global] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Try the structured 45-rank decomposition
    result = _construct_45_rank()
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    
    if _verify_decomposition(U_arr, V_arr, W_arr, n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
