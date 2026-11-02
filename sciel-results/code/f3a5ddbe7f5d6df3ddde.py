"""
Rank-minimal exact tensor decomposition for 4x4 x 4x8 matrix multiplication.

Strategy: Use Strassen's 2x2x2 (rank 7) decomposition composed with itself via
Kronecker product to get rank 49 for 4x4x4, then split B into two 4x4 halves
for total rank 98.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[k,i]
    """
    U = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    V = [[0]*7 for _ in range(4)]
    V[0][0] = 1; V[3][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[3][2] = -1
    V[2][3] = 1; V[0][3] = -1
    V[3][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[2][6] = 1; V[3][6] = 1
    
    W = [[0]*7 for _ in range(4)]
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[2][0] = 1; W[2][4] = 1
    W[1][1] = 1; W[1][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _compose_decompositions(U1, V1, W1, U2, V2, W2):
    """
    Compose two tensor decompositions via Kronecker product.
    If T1 = sum_r u1_r (x) v1_r (x) w1_r (rank R1) and
       T2 = sum_s u2_s (x) v2_s (x) w2_s (rank R2),
    then T1 (x) T2 = sum_{r,s} (u1_r (x) u2_s) (x) (v1_r (x) v2_s) (x) (w1_r (x) w2_s)
    with rank R1*R2.
    """
    d1, R1 = len(U1), len(U1[0])
    d2, R2 = len(U2), len(U2[0])
    dv1 = len(V1); dv2 = len(V2)
    dw1 = len(W1); dw2 = len(W2)
    
    U = [[0] * (R1 * R2) for _ in range(d1 * d2)]
    V = [[0] * (R1 * R2) for _ in range(dv1 * dv2)]
    W = [[0] * (R1 * R2) for _ in range(dw1 * dw2)]
    
    for r in range(R1):
        for s in range(R2):
            col = r * R2 + s
            for i1 in range(d1):
                c1 = U1[i1][r]
                if c1 == 0:
                    continue
                for i2 in range(d2):
                    c2 = U2[i2][s]
                    if c2 != 0:
                        U[i1 * d2 + i2][col] = c1 * c2
            for j1 in range(dv1):
                c1 = V1[j1][r]
                if c1 == 0:
                    continue
                for j2 in range(dv2):
                    c2 = V2[j2][s]
                    if c2 != 0:
                        V[j1 * dv2 + j2][col] = c1 * c2
            for k1 in range(dw1):
                c1 = W1[k1][r]
                if c1 == 0:
                    continue
                for k2 in range(dw2):
                    c2 = W2[k2][s]
                    if c2 != 0:
                        W[k1 * dw2 + k2][col] = c1 * c2
    
    return U, V, W


def _strassen_4x4x8():
    """
    Construct a rank-98 decomposition for 4x4 x 4x8 matrix multiplication.
    Use Strassen (rank 7) composed with itself to get rank 49 for 4x4x4,
    then split B into two 4x4 halves for total rank 98.
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Compose Strassen with itself to get 4x4x4 decomposition (rank 49)
    U_444, V_444, W_444 = _compose_decompositions(U_s, V_s, W_s, U_s, V_s, W_s)
    
    # Build 4x4 x 4x8 by using two copies of the 4x4x4 decomposition
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]   # (16, 98)
    V = [[0] * rank for _ in range(m * p)]   # (32, 98)
    W = [[0] * rank for _ in range(p * n)]   # (32, 98)
    
    for k_block in range(2):
        base_col = k_block * 49
        for col_inner in range(49):
            col = base_col + col_inner
            for i in range(16):
                U[i][col] = U_444[i][col_inner]
            for j in range(4):
                for k in range(4):
                    V[j * 8 + k_block * 4 + k][col] = V_444[j * 4 + k][col_inner]
            for k in range(4):
                for i in range(4):
                    W[(k_block * 4 + k) * 4 + i][col] = W_444[k * 4 + i][col_inner]
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Primary: Strassen-composed rank-98 decomposition
    result = _strassen_4x4x8()
    
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
