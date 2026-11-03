"""Strassen-enhanced tensor decomposition for 3x5 by 5x5 matrix multiplication."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

def _verify(U, V, W, n, m, p, rank, seed=42):
    """Verify tensor decomposition correctness using multiple random tests."""
    rng = np.random.RandomState(seed)
    for _ in range(3):
        A_test = rng.rand(n, m)
        B_test = rng.rand(m, p)
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    for j in range(m):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = total
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _strassen_2x2x2(U, V, W, u_rows, v_rows, w_rows, col):
    """
    Apply Strassen's 7-term decomposition for 2x2x2 block multiplication.
    u_rows = [u11, u12, u21, u22] (U row indices for A[0,0], A[0,1], A[1,0], A[1,1])
    v_rows = [v11, v12, v21, v22] (V row indices for B[0,0], B[0,1], B[1,0], B[1,1])
    w_rows = [w11, w12, w21, w22] (W row indices for C[0,0], C[0,1], C[1,0], C[1,1])
    Returns next col index.
    """
    u11, u12, u21, u22 = u_rows
    v11, v12, v21, v22 = v_rows
    w11, w12, w21, w22 = w_rows
    
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    U[u11, col] += 1; U[u22, col] += 1
    V[v11, col] += 1; V[v22, col] += 1
    W[w11, col] += 1; W[w22, col] += 1
    col += 1
    
    # M2 = (A21+A22)B11 -> contributes to C21, C22
    U[u21, col] += 1; U[u22, col] += 1
    V[v11, col] += 1
    W[w21, col] += 1; W[w22, col] += 1
    col += 1
    
    # M3 = A11(B12-B22) -> contributes to C12, C22
    U[u11, col] += 1
    V[v12, col] += 1; V[v22, col] -= 1
    W[w12, col] += 1; W[w22, col] += 1
    col += 1
    
    # M4 = A22(B21-B11) -> contributes to C21, C22
    U[u22, col] += 1
    V[v21, col] += 1; V[v11, col] -= 1
    W[w21, col] += 1; W[w22, col] += 1
    col += 1
    
    # M5 = (A11+A12)B22 -> contributes to C11, C12
    U[u11, col] += 1; U[u12, col] += 1
    V[v22, col] += 1
    W[w11, col] += 1; W[w12, col] += 1
    col += 1
    
    # M6 = (A21-A11)(B11+B12) -> contributes to C21, C22
    U[u21, col] += 1; U[u11, col] -= 1
    V[v11, col] += 1; V[v12, col] += 1
    W[w21, col] += 1; W[w22, col] += 1
    col += 1
    
    # M7 = (A12-A22)(B21+B22) -> contributes to C11, C12
    U[u12, col] += 1; U[u22, col] -= 1
    V[v21, col] += 1; V[v22, col] += 1
    W[w11, col] += 1; W[w12, col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    deadline = start_time + 115

    # Strategy: Partition the middle dimension (5) into 2+3.
    # Apply Strassen to 2x2x2 sub-problems where beneficial.
    #
    # A (3x5) = [A_L (3x2) | A_R (3x3)]
    # B (5x5) = [B_L (2x5); B_R (3x5)]
    # C = A_L * B_L + A_R * B_R
    #
    # Partition:
    # A: rows 3=1+2, cols 5=2+3
    # B: rows 5=2+3, cols 5=2+3
    #
    # Block structure:
    # A = [[a11(1x2), a12(1x3)],
    #      [a21(2x2), a22(2x3)]]
    # B = [[b11(2x2), b12(2x3)],
    #      [b21(3x2), b22(3x3)]]
    #
    # C = A*B:
    # C[0,0:2]   = a11*b11 + a12*b21
    # C[0,2:5]   = a11*b12 + a12*b22
    # C[1:3,0:2] = a21*b11 + a22*b21
    # C[1:3,2:5] = a21*b12 + a22*b22
    #
    # Schoolbook counts:
    # a11*b11: 4, a11*b12: 6, a12*b21: 6, a12*b22: 9
    # a21*b11: 8, a21*b12: 12, a22*b21: 12, a22*b22: 18
    # Total: 75
    #
    # Strassen on a21*b11 (2x2x2): saves 1 -> total 74
    #
    # Additionally, we can try to find savings in the 2x3x3 and 3x2x2 blocks
    # by examining if any sub-structures allow rank reduction.
    # For now, we use the known rank-74 decomposition.
    
    rank = 74
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # a11*b11: (1x2)(2x2) -> 4 terms (schoolbook, no saving possible)
    for i in [0]:
        for j in [0, 1]:
            for k in [0, 1]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 4
    
    # a11*b12: (1x2)(2x3) -> 6 terms (schoolbook)
    for i in [0]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 10
    
    # a12*b21: (1x3)(3x2) -> 6 terms (schoolbook)
    for i in [0]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 16
    
    # a12*b22: (1x3)(3x3) -> 9 terms (schoolbook)
    for i in [0]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 25
    
    # a21*b11: (2x2)(2x2) -> Strassen: 7 terms
    # U rows: 5=A[1,0], 6=A[1,1], 7=A[2,0], 8=A[2,1]
    # V rows: 0=B[0,0], 1=B[0,1], 4=B[1,0], 5=B[1,1]
    # W rows: 5=C[1,0], 6=C[1,1], 7=C[2,0], 8=C[2,1]
    col = _strassen_2x2x2(
        U, V, W,
        u_rows=[5, 6, 7, 8],
        v_rows=[0, 1, 4, 5],
        w_rows=[5, 6, 7, 8],
        col=col
    )
    # col = 32
    
    # a21*b12: (2x2)(2x3) -> 12 terms (schoolbook)
    for i in [1, 2]:
        for j in [0, 1]:
            for k in [2, 3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 44
    
    # a22*b21: (2x3)(3x2) -> 12 terms (schoolbook)
    for i in [1, 2]:
        for j in [2, 3, 4]:
            for k in [0, 1]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 56
    
    # a22*b22: (2x3)(3x3) -> 18 terms (schoolbook)
    for i in [1, 2]:
        for j in [2, 3, 4]:
            for k in [2, 3, 4]:
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    # col = 74
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition is correct
    if _verify(U, V, W, n, m, p, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
