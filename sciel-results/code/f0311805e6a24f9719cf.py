"""Combined program for matrix multiplication 2x3x8 tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _check_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _check_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast float-based check."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2."""
    # Strassen's algorithm for 2x2 matrices
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    
    # U (2x2), V (2x2), W (2x2)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U[i,j], V[j,k], W[k,i]
    
    U = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]]
    V = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]]
    W = [[0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0], [0,0,0,0,0,0,0]]
    
    # M1
    U[0,0][0] = 1; U[1,1][0] = 1
    V[0,0][0] = 1; V[1,1][0] = 1
    W[0,0][0] = 1; W[1,1][0] = 1; W[0,1][0] = -1; W[1,0][0] = -1
    
    # M2
    U[1,0][1] = 1; U[1,1][1] = 1
    V[0,0][1] = 1
    W[1,0][1] = 1
    
    # M3
    U[0,0][2] = 1
    V[0,1][2] = 1; V[1,1][2] = -1
    W[0,0][2] = 1
    
    # M4
    U[1,1][3] = 1
    V[1,0][3] = 1; V[0,0][3] = -1
    W[1,1][3] = 1
    
    # M5
    U[0,0][4] = 1; U[0,1][4] = 1
    V[1,1][4] = 1
    W[0,1][4] = 1
    
    # M6
    U[1,0][5] = 1; U[0,0][5] = -1
    V[0,0][5] = 1; V[0,1][5] = 1
    W[1,0][5] = 1
    
    # M7
    U[0,1][6] = 1; U[1,1][6] = -1
    V[1,0][6] = 1; V[1,1][6] = 1
    W[0,1][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Use the block decomposition:
    # M_{2,3,8} = M_{2,1,8} + M_{2,2,8}
    # Rank(M_{2,1,8}) = 16 (schoolbook)
    # Rank(M_{2,2,8}) = 2 * Rank(M_{2,2,4}) = 2 * (2 * Rank(M_{2,2,2})) = 4 * 7 = 28
    # Total Rank = 16 + 28 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: M_{2,1,8} (j=0)
    # i in {0,1}, j=0, k in {0..7}
    for i in range(n):
        for k in range(p):
            U[i*m + 0][col] = 1
            V[0*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Part 2: M_{2,2,8} (j=1,2)
    # Split k into two blocks of 4: k1 in {0..3}, k2 in {4..7}
    # For each block, use 2x2x2 Strassen twice?
    # M_{2,2,4} = M_{2,2,2} (k=0,1) + M_{2,2,2} (k=2,3)
    # So M_{2,2,8} has 4 blocks of 2x2x2.
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    # Map local indices to global indices
    # Local i: 0,1 -> Global i: 0,1
    # Local j: 0,1 -> Global j: 1,2 (since j=0 was used in Part 1)
    # Local k: 0,1 -> Global k: depends on block
    
    for block in range(4):
        k_offset = block * 2
        for c_local in range(7):
            c_global = col + c_local
            for i_local in range(2):
                i_global = i_local
                for j_local in range(2):
                    j_global = 1 + j_local
                    U[i_global*m + j_global][c_global] = strassen_U[i_local, j_local][c_local]
            for j_local in range(2):
                j_global = 1 + j_local
                for k_local in range(2):
                    k_global = k_offset + k_local
                    V[j_global*p + k_global][c_global] = strassen_V[j_local, k_local][c_local]
            for k_local in range(2):
                k_global = k_offset + k_local
                for i_local in range(2):
                    i_global = i_local
                    W[k_global*n + i_global][c_global] = strassen_W[k_local, i_local][c_local]
        col += 7
        
    # Verify
    if _check_fast(U, V, W, n, m, p):
        if _check_exact(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
