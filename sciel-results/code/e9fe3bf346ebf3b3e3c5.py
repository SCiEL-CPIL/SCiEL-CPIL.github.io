"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 1, 1, 0, 1]
    ])
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using Kronecker product of 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    n = 4
    m = 4
    p = 4
    
    U4 = np.zeros((n*m, rank))
    V4 = np.zeros((m*p, rank))
    W4 = np.zeros((p*n, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U4[i, r] = U2[i1, r1] * U2[i2, r2]
                    
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            V4[j, r] = V2[j1, r1] * V2[j2, r2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    # W index is k*n + i
                                    W4[k*n + i, r] = W2[k1*2 + i1, r1] * W2[k2*2 + i2, r2]
                                    
    return U4, V4, W4

def _build_4x4x1_schoolbook():
    """Build 4x4x1 decomposition (dot product) with rank 16."""
    n = 4
    m = 4
    p = 1
    rank = 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m + j, col] = 1
            V[j*p + 0, col] = 1
            W[0*n + i, col] = 1
            col += 1
            
    return U, V, W

def _build_4x4x9_decomposition():
    """Construct the full 4x4x9 decomposition."""
    n, m, p = 4, 4, 9
    U4, V4, W4 = _build_4x4x4_strassen()
    U1, V1, W1 = _build_4x4x1_schoolbook()
    
    rank_total = 49 + 49 + 16
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # Part 1: Columns 0-3 of B (B1) -> Columns 0-3 of C (C1)
    # U is same for all parts of A
    U[:, 0:49] = U4
    
    # V for B1: k in 0..3
    for j in range(m):
        for k_local in range(4):
            V[j*p + k_local, 0:49] = V4[j, :]
            
    # W for C1: k in 0..3
    for k_local in range(4):
        for i in range(n):
            W[k_local*n + i, 0:49] = W4[k_local*n + i, :]
            
    # Part 2: Columns 4-7 of B (B2) -> Columns 4-7 of C (C2)
    U[:, 49:98] = U4
    
    # V for B2: k in 4..7
    for j in range(m):
        for k_local in range(4):
            V[j*p + (4 + k_local), 49:98] = V4[j, :]
            
    # W for C2: k in 4..7
    for k_local in range(4):
        for i in range(n):
            W[(4 + k_local)*n + i, 49:98] = W4[k_local*n + i, :]
            
    # Part 3: Column 8 of B (B3) -> Column 8 of C (C3)
    U[:, 98:114] = U1
    
    # V for B3: k=8
    for j in range(m):
        V[j*p + 8, 98:114] = V1[j, :]
        
    # W for C3: k=8
    for i in range(n):
        W[8*n + i, 98:114] = W1[i, :]
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_4x4x9_decomposition()
    
    # Verify
    T_reconstructed = np.zeros((n, m, p))
    rank = U.shape[1]
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    
    T_expected = np.ones((n, m, p))
    if not np.allclose(T_reconstructed, T_expected, atol=1e-9):
        # Fallback to schoolbook if verification fails
        U_s, V_s, W_s, _ = _schoolbook_decomposition(n, m, p)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return U, V, W, rank
# EVOLVE-BLOCK-END
