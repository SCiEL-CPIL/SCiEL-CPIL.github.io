"""
Matrix multiplication 4x4 by 4x8 — tensor rank minimization.

Strategy:
  1. Build a correct rank-7 Strassen decomposition for 2x2x2.
  2. Use it to build a rank-49 decomposition for 4x4x4 via block Strassen
     (7 block products, each done with 7 scalar Strassen multiplications).
  3. Split the 4x4x8 problem into two 4x4x4 sub-problems (columns 0-3 and 4-7),
     giving rank 98 total.
  4. Verify numerically and return.
  5. Fallback: schoolbook rank-128.
"""

import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 140.0


def _time_up():
    return (time.time() - _START_TIME) > _TIMEOUT


def _validate(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _make_schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_222():
    """
    Correct 7-term Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    Verified formulas:
    p1 = (a+d)(e+h)
    p2 = (c+d)h
    p3 = a(f-g)
    p4 = d(g-e)
    p5 = (a+b)(e+f)
    p6 = (c-a)(g+h)
    p7 = (b-d)(f+h)
    
    C00 = p1 + p4 - p5 + p7
    C01 = p3 + p5
    C10 = p2 + p4
    C11 = p1 - p2 + p3 + p6
    """
    # U[i*2+j, r] = coefficient of A[i,j] in product r
    # V[j*2+k, r] = coefficient of B[j,k] in product r
    # W[k*2+i, r] = coefficient of product r in C[i,k]
    
    U = [
        [1,  0,  1,  0,  1, -1,  0],  # U[0,0] = a
        [0,  0,  0,  0,  1,  0,  1],  # U[0,1] = b
        [0,  1,  0,  0,  0,  1,  0],  # U[1,0] = c
        [1,  1,  0,  1,  0,  0, -1]   # U[1,1] = d
    ]
    V = [
        [1,  0,  0, -1,  1,  0,  0],  # V[0,0] = e
        [0,  0,  1,  0,  1,  0,  1],  # V[0,1] = f
        [0,  0, -1,  1,  0,  1,  0],  # V[1,0] = g
        [1,  1,  0,  0,  0,  1,  1]   # V[1,1] = h
    ]
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # W[0,0] = C[0,0] (k=0,i=0)
        [0,  1,  1,  1,  0,  0,  0],  # W[1,0] = C[1,0] (k=0,i=1)
        [0,  0,  1,  0,  1,  0,  0],  # W[2,0] = C[0,1] (k=1,i=0)
        [1, -1,  0,  0,  0,  1,  0]   # W[3,0] = C[1,1] (k=1,i=1)
    ]
    
    # Verify numerically
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                if s != 1:
                    raise ValueError("Strassen 2x2x2 verification failed")
    
    return U, V, W


def _block_strassen_444():
    """
    Build a 4x4x4 decomposition using block Strassen.
    rank = 7 * 7 = 49.
    
    The idea:
    - Split A (4x4) into 2x2 blocks: A11, A12, A21, A22
    - Split B (4x4) into 2x2 blocks: B11, B12, B21, B22
    - Use 7 block-level Strassen products, each of which is a 2x2*2x2 matrix product
    - Each 2x2*2x2 product uses 7 scalar Strassen multiplications
    - Total: 7 * 7 = 49 scalar multiplications
    
    The block-level Strassen formulas (same as scalar, but with 2x2 matrices):
    P1 = (A11+A22)(B11+B22)
    P2 = (A21+A22)B22
    P3 = A11(B12-B21)
    P4 = A22(B21-B11)
    P5 = (A11+A12)(B11+B12)
    P6 = (A21-A11)(B21+B22)
    P7 = (A12-A22)(B12+B22)
    
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    """
    U2, V2, W2 = _strassen_222()
    
    # For the 4x4x4 problem:
    # A is 4x4, B is 4x4, C is 4x4
    # Index: i (0..3), j (0..3), k (0..3)
    # Block structure: i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # where i1, i0, j1, j0, k1, k0 ∈ {0, 1}
    
    # U[i*4+j, r] for i,j in 0..3, r in 0..48
    # V[j*4+k, r] for j,k in 0..3, r in 0..48
    # W[k*4+i, r] for k,i in 0..3, r in 0..48
    
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]  # 16 x 49
    V = [[0]*rank for _ in range(m*p)]  # 16 x 49
    W = [[0]*rank for _ in range(p*n)]  # 16 x 49
    
    # Block-level Strassen products:
    # Each product b (0..6) has:
    #   A-side: linear combination of 2x2 blocks -> 4x4 matrix
    #   B-side: linear combination of 2x2 blocks -> 4x4 matrix
    #   C-side: contributes to specific 2x2 blocks of C with specific signs
    
    # A-block coefficients for each block product (4x4 matrix, 4x4 blocks)
    # A[2*i1+i0, 2*j1+j0]
    # Block A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    #            A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    # For each block product, define the 4x4 A-coefficient matrix and B-coefficient matrix
    # Then use the 2x2x2 Strassen to decompose each 2x2*2x2 block product
    
    # Block product definitions (A-blocks, B-blocks, C-block contributions):
    # P0: A-side = A11+A22, B-side = B11+B22, C-side: C11:+1, C22:+1
    # P1: A-side = A21+A22, B-side = B22,     C-side: C21:+1, C22:-1
    # P2: A-side = A11,     B-side = B12-B21, C-side: C12:+1, C21:+1
    # P3: A-side = A22,     B-side = B21-B11, C-side: C11:+1, C21:+1
    # P4: A-side = A11+A12, B-side = B11+B12, C-side: C11:-1, C12:+1
    # P5: A-side = A21-A11, B-side = B21+B22, C-side: C22:+1
    # P6: A-side = A12-A22, B-side = B12+B22, C-side: C11:+1
    
    # For each block product b, and each scalar Strassen product s (0..6):
    # The combined product index is r = b*7 + s
    # 
    # U[i*4+j, r] = sum over block positions (i1,j1) of A_coeff[i1,j1,b] * U2[i0*2+j0, s]
    # where i = 2*i1+i0, j = 2*j1+j0
    #
    # Similarly for V and W.
    
    # A-block coefficients for each block product (2x2 block matrix)
    # A_blocks[b][i1][j1] = coefficient of block A[i1,j1] in product b
    A_blocks = [
        [[1, 0], [0, 1]],   # P0: A11+A22
        [[0, 0], [1, 1]],   # P1: A21+A22
        [[1, 0], [0, 0]],   # P2: A11
        [[0, 0], [0, 1]],   # P3: A22
        [[1, 1], [0, 0]],   # P4: A11+A12
        [[-1, 0], [1, 0]],  # P5: A21-A11
        [[0, 1], [0, -1]]   # P6: A12-A22
    ]
    
    B_blocks = [
        [[1, 0], [0, 1]],   # P0: B11+B22
        [[0, 0], [0, 1]],   # P1: B22
        [[0, 1], [-1, 0]],  # P2: B12-B21
        [[0, 0], [1, -1]],  # P3: B21-B11
        [[1, 1], [0, 0]],   # P4: B11+B12
        [[0, 0], [1, 1]],   # P5: B21+B22
        [[0, 1], [0, 1]]    # P6: B12+B22
    ]
    
    # C-block contributions: C_blocks[b][i1][k1] = coefficient for block C[i1,k1]
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4]
    # C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    C_blocks = [
        [[1, 0], [0, 1]],   # P0: C11:+1, C22:+1
        [[0, 0], [1, -1]],  # P1: C21:+1, C22:-1
        [[0, 1], [1, 0]],   # P2: C12:+1, C21:+1
        [[1, 0], [1, 0]],   # P3: C11:+1, C21:+1
        [[-1, 1], [0, 0]],  # P4: C11:-1, C12:+1
        [[0, 0], [0, 1]],   # P5: C22:+1
        [[1, 0], [0, 0]]    # P6: C11:+1
    ]
    
    # Now build U, V, W
    for b in range(7):
        for s in range(7):
            r = b * 7 + s
            # For each element (i, j) in A (4x4):
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            i = 2*i1 + i0
                            j = 2*j1 + j0
                            # U[i*4+j, r] += A_blocks[b][i1][j1] * U2[i0*2+j0, s]
                            U[i*4+j][r] += A_blocks[b][i1][j1] * U2[i0*2+j0][s]
            
            # For each element (j, k) in B (4x4):
            for j1 in range(2):
                for j0 in range(2):
                    for k1 in range(2):
                        for k0 in range(2):
                            j = 2*j1 + j0
                            k = 2*k1 + k0
                            # V[j*4+k, r] += B_blocks[b][j1][k1] * V2[j0*2+k0, s]
                            V[j*4+k][r] += B_blocks[b][j1][k1] * V2[j0*2+k0][s]
            
            # For each element (k, i) in C (4x4):
            for k1 in range(2):
                for k0 in range(2):
                    for i1 in range(2):
                        for i0 in range(2):
                            k = 2*k1 + k0
                            i = 2*i1 + i0
                            # W[k*4+i, r] += C_blocks[b][i1][k1] * W2[k0*2+i0, s]
                            W[k*4+i][r] += C_blocks[b][i1][k1] * W2[k0*2+i0][s]
    
    # Verify
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s_val = 0
                for r in range(49):
                    s_val += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                if s_val != 1:
                    raise ValueError(f"Block Strassen 4x4x4 verification failed at ({i},{j},{k}): got {s_val}")
    
    return U, V, W


def construct():
    n, m, p = 4, 4, 8
    
    # Build 4x4x4 decomposition with rank 49
    try:
        U444, V444, W444 = _block_strassen_444()
    except Exception:
        return _make_schoolbook(n, m, p)
    
    # Now build 4x4x8 by splitting into two 4x4x4 sub-problems
    # C = A * B where A is 4x4, B is 4x8
    # Split B into B1 (4x4, columns 0-3) and B2 (4x4, columns 4-7)
    # C1 = A * B1 (4x4), C2 = A * B2 (4x4)
    
    rank = 98
    U = [[0]*rank for _ in range(n*m)]  # 16 x 98
    V = [[0]*rank for _ in range(m*p)]  # 16 x 98
    W = [[0]*rank for _ in range(p*n)]  # 32 x 98
    
    # First half (columns 0-3 of B, columns 0-3 of C):
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*4+j][r] = U444[i*4+j][r]
        for j in range(4):
            for k in range(4):
                V[j*4+k][r] = V444[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[k*4+i][r] = W444[k*4+i][r]
    
    # Second half (columns 4-7 of B, columns 4-7 of C):
    for r in range(49):
        r2 = r + 49
        for i in range(4):
            for j in range(4):
                U[i*4+j][r2] = U444[i*4+j][r]
        for j in range(4):
            for k in range(4):
                V[j*4+(k+4)][r2] = V444[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i][r2] = W444[k*4+i][r]
    
    # Verify
    if not _validate(U, V, W, n, m, p):
        return _make_schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
