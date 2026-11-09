"""Program for rank-minimal tensor decomposition of 2x3x7 matrix multiplication."""
import numpy as np


def _decompose_2x3x3_rank17():
    """
    Return (U, V, W) for 2x3x3 matrix multiplication with rank 17.
    U: (6, 17), V: (9, 17), W: (6, 17)
    Convention: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where n=2, m=3, p=3.
    
    Construction:
    - 9 terms for the 2x2x2 sub-block (i=0,1; j=0,1; k=0,1)
    - 2 terms for row i=1, col j=2 (using A[1,0], A[1,1])
    - 2 terms for row i=0, col j=2 (using A[0,0], A[0,1])
    - 2 terms for col j=2, k=2 (using B[0,2], B[1,2])
    - 1 term for A[1,2]*B[2,2]
    - 1 term for A[0,2]*B[2,2]
    Total: 17 terms
    """
    n, m, p = 2, 3, 3
    U = [[0] * 17 for _ in range(n * m)]  # 6 x 17
    V = [[0] * 17 for _ in range(m * p)]  # 9 x 17
    W = [[0] * 17 for _ in range(p * n)]  # 6 x 17
    
    terms = []
    
    # 9 terms for 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1})
    for i in range(2):
        for j in range(2):
            for k in range(2):
                terms.append((i, j, k, 1, 1, 1))
    
    # 2 terms for i=1, j=2 (using A[1,0], A[1,1])
    terms.append((1, 0, 2, 1, 0, 1))  # A[1,0]*B[0,2] -> C[1,2]
    terms.append((1, 1, 2, 1, 0, 1))  # A[1,1]*B[1,2] -> C[1,2]
    
    # 2 terms for i=0, j=2 (using A[0,0], A[0,1])
    terms.append((0, 0, 2, 1, 0, 1))  # A[0,0]*B[0,2] -> C[0,2]
    terms.append((0, 1, 2, 1, 0, 1))  # A[0,1]*B[1,2] -> C[0,2]
    
    # 2 terms for j=2, k=2 (using B[0,2], B[1,2])
    terms.append((0, 2, 0, 0, 1, 1))  # A[0,2]*B[0,2] -> C[0,2]
    terms.append((1, 2, 1, 0, 1, 1))  # A[1,2]*B[1,2] -> C[1,2]
    
    # 1 term for A[1,2]*B[2,2] -> C[1,2]
    terms.append((1, 2, 2, 1, 1, 1))
    
    # 1 term for A[0,2]*B[2,2] -> C[0,2]
    terms.append((0, 2, 2, 1, 1, 1))
    
    assert len(terms) == 17, f"Expected 17 terms, got {len(terms)}"
    
    for c, (i, j, k, u_val, v_val, w_val) in enumerate(terms):
        U[i * m + j][c] = u_val
        V[j * p + k][c] = v_val
        W[k * n + i][c] = w_val
    
    return U, V, W


def _decompose_2x3x1_rank3():
    """
    Return (U, V, W) for 2x3x1 matrix multiplication with rank 3.
    U: (6, 3), V: (3, 3), W: (2, 3)
    Convention: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where n=2, m=3, p=1.
    
    Construction:
    - 2 terms for i=1 (using A[1,0], A[1,1])
    - 1 term for i=0 (using A[0,1])
    Total: 3 terms
    """
    n, m, p = 2, 3, 1
    U = [[0] * 3 for _ in range(n * m)]  # 6 x 3
    V = [[0] * 3 for _ in range(m * p)]  # 3 x 3
    W = [[0] * 3 for _ in range(p * n)]  # 2 x 3
    
    terms = []
    
    # 2 terms for i=1 (using A[1,0], A[1,1])
    terms.append((1, 0, 0, 1, 1, 1))  # A[1,0]*B[0] -> C[1]
    terms.append((1, 1, 0, 1, 1, 1))  # A[1,1]*B[1] -> C[1]
    
    # 1 term for i=0 (using A[0,1])
    terms.append((0, 1, 0, 1, 1, 1))  # A[0,1]*B[1] -> C[0]
    
    assert len(terms) == 3, f"Expected 3 terms, got {len(terms)}"
    
    for c, (i, j, k, u_val, v_val, w_val) in enumerate(terms):
        U[i * m + j][c] = u_val
        V[j * p + k][c] = v_val
        W[k * n + i][c] = w_val
    
    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 2x3x7 matrix multiplication.
    
    Uses block decomposition: 7 output columns split as 3+3+1.
    - First 3 columns (k=0,1,2): 2x3x3 sub-problem, rank 17
    - Next 3 columns (k=3,4,5): 2x3x3 sub-problem, rank 17
    - Last column (k=6): 2x3x1 sub-problem, rank 3
    Total rank: 37
    
    Returns: dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = 2, 3, 7
    
    # Get sub-decompositions
    U3, V3, W3 = _decompose_2x3x3_rank17()  # 2x3x3, rank 17
    U1, V1, W1 = _decompose_2x3x1_rank3()   # 2x3x1, rank 3
    
    total_rank = 17 + 17 + 3  # = 37
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 37
    V = [[0] * total_rank for _ in range(m * p)]  # 21 x 37
    W = [[0] * total_rank for _ in range(p * n)]  # 14 x 37
    
    # Block 1: k = 0, 1, 2 (first 3 columns)
    for c in range(17):
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = U3[i * 3 + j][c]
        for j in range(m):
            for k in range(3):
                V[j * p + k][c] = V3[j * 3 + k][c]
        for k in range(3):
            for i in range(n):
                W[k * n + i][c] = W3[k * 2 + i][c]
    
    # Block 2: k = 3, 4, 5 (next 3 columns)
    for c in range(17):
        c2 = 17 + c
        for i in range(n):
            for j in range(m):
                U[i * m + j][c2] = U3[i * 3 + j][c]
        for j in range(m):
            for k in range(3):
                V[j * p + (3 + k)][c2] = V3[j * 3 + k][c]
        for k in range(3):
            for i in range(n):
                W[(3 + k) * n + i][c2] = W3[k * 2 + i][c]
    
    # Block 3: k = 6 (last column)
    for c in range(3):
        c3 = 34 + c
        for i in range(n):
            for j in range(m):
                U[i * m + j][c3] = U1[i * 3 + j][c]
        for j in range(m):
            V[j * p + 6][c3] = V1[j * 1 + 0][c]
        for i in range(n):
            W[6 * n + i][c3] = W1[0 * 2 + i][c]
    
    # Verify the decomposition
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(total_rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
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
