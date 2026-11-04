"""Matrix multiplication 4x5x5 with Strassen-based core decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix.
    
    Strategy:
    1. Split the 4x5x5 problem into a 4x4x4 core and fringe terms.
    2. The core A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4] is decomposed 
       into 8 sub-problems of size 2x2x2.
       Each 2x2x2 problem is solved using Strassen's algorithm (rank 7).
       Total core rank = 8 * 7 = 56.
    3. The fringe terms are:
       - Terms where j=4: A[i,4] * B[4,k] for i in 0..3, k in 0..4.
         This is a 4x1x5 tensor, rank 20 (schoolbook).
       - Terms where k=4 and j<4: A[i,j] * B[j,4] for i in 0..3, j in 0..3.
         This is a 4x4x1 tensor, rank 16 (schoolbook).
    4. Total rank = 56 + 20 + 16 = 92.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Total rank
    rank_core = 56
    rank_fringe_j = 20
    rank_fringe_k = 16
    total_rank = rank_core + rank_fringe_j + rank_fringe_k
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for a 2x2x2 sub-block
    # A indices: i1, i2 (rows), j1, j2 (cols)
    # B indices: j1, j2 (rows), k1, k2 (cols)
    # C indices: i1, i2 (rows), k1, k2 (cols)
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # p1 = (a+d)(e+h)
        # U: A[i1,j1] + A[i2,j2]
        # V: B[j1,k1] + B[j2,k2]
        # W: C[i1,k1] += 1, C[i1,k2] += 1, C[i2,k1] += 1, C[i2,k2] -= 1
        U[i1*m + j1][col] += 1
        U[i2*m + j2][col] += 1
        V[j1*p + k1][col] += 1
        V[j2*p + k2][col] += 1
        W[k1*n + i1][col] += 1
        W[k2*n + i1][col] += 1
        W[k1*n + i2][col] += 1
        W[k2*n + i2][col] -= 1
        col += 1
        
        # p2 = (c+d)e
        # U: A[i2,j1] + A[i2,j2]
        # V: B[j1,k1]
        # W: C[i2,k1] += 1, C[i2,k2] += 1
        U[i2*m + j1][col] += 1
        U[i2*m + j2][col] += 1
        V[j1*p + k1][col] += 1
        W[k1*n + i2][col] += 1
        W[k2*n + i2][col] += 1
        col += 1
        
        # p3 = a(f-h)
        # U: A[i1,j1]
        # V: B[j1,k2] - B[j2,k2]
        # W: C[i1,k2] += 1, C[i2,k2] += 1
        U[i1*m + j1][col] += 1
        V[j1*p + k2][col] += 1
        V[j2*p + k2][col] -= 1
        W[k2*n + i1][col] += 1
        W[k2*n + i2][col] += 1
        col += 1
        
        # p4 = d(g-e)
        # U: A[i2,j2]
        # V: B[j2,k1] - B[j1,k1]
        # W: C[i1,k1] += 1, C[i2,k1] += 1
        U[i2*m + j2][col] += 1
        V[j2*p + k1][col] += 1
        V[j1*p + k1][col] -= 1
        W[k1*n + i1][col] += 1
        W[k1*n + i2][col] += 1
        col += 1
        
        # p5 = (a+b)h
        # U: A[i1,j1] + A[i1,j2]
        # V: B[j2,k2]
        # W: C[i1,k1] += 1, C[i1,k2] += 1
        U[i1*m + j1][col] += 1
        U[i1*m + j2][col] += 1
        V[j2*p + k2][col] += 1
        W[k1*n + i1][col] += 1
        W[k2*n + i1][col] += 1
        col += 1
        
        # p6 = (c-a)(e+f)
        # U: A[i2,j1] - A[i1,j1]
        # V: B[j1,k1] + B[j1,k2]
        # W: C[i2,k1] += 1, C[i2,k2] += 1
        U[i2*m + j1][col] += 1
        U[i1*m + j1][col] -= 1
        V[j1*p + k1][col] += 1
        V[j1*p + k2][col] += 1
        W[k1*n + i2][col] += 1
        W[k2*n + i2][col] += 1
        col += 1
        
        # p7 = (b-d)(g+h)
        # U: A[i1,j2] - A[i2,j2]
        # V: B[j2,k1] + B[j2,k2]
        # W: C[i1,k1] += 1, C[i2,k2] += 1
        U[i1*m + j2][col] += 1
        U[i2*m + j2][col] -= 1
        V[j2*p + k1][col] += 1
        V[j2*p + k2][col] += 1
        W[k1*n + i1][col] += 1
        W[k2*n + i2][col] += 1
        col += 1

    # 1. Core 4x4x4 decomposition
    # Split 4x4 into four 2x2 blocks
    # A blocks: (0,1)x(0,1), (0,1)x(2,3), (2,3)x(0,1), (2,3)x(2,3)
    # B blocks: (0,1)x(0,1), (0,1)x(2,3), (2,3)x(0,1), (2,3)x(2,3)
    
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 1, 2, 3, 0, 1)
    
    # C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 1, 2, 3, 2, 3)
    
    # C[2:4, 0:2] = A[2:4, 0:2] * B[0:2, 0:2] + A[2:4, 2:4] * B[2:4, 0:2]
    # Block 5: A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    add_strassen(2, 3, 0, 1, 0, 1)
    # Block 6: A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]
    add_strassen(2, 3, 2, 3, 0, 1)
    
    # C[2:4, 2:4] = A[2:4, 0:2] * B[0:2, 2:4] + A[2:4, 2:4] * B[2:4, 2:4]
    # Block 7: A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    add_strassen(2, 3, 0, 1, 2, 3)
    # Block 8: A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    add_strassen(2, 3, 2, 3, 2, 3)
    
    # 2. Fringe: j=4 terms
    # C[i,k] += A[i,4] * B[4,k]
    for i in range(4):
        for k in range(5):
            U[i*m + 4][col] = 1
            V[4*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # 3. Fringe: k=4 terms (j<4)
    # C[i,4] += A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
