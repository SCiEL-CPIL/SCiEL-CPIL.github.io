import time
import numpy as np

def construct():
    """
    Returns an exact tensor decomposition for multiplying a 2x3 matrix 
    by a 3x5 matrix. Uses block decomposition with Strassen for 2x2x2
    subproblems and schoolbook for smaller blocks.
    
    Decomposition:
    - Block 1: j=0, k=0..4 (2x1x5): rank 10 (schoolbook)
    - Block 2a: j=1,2, k=0,1 (2x2x2): rank 7 (Strassen)
    - Block 2b: j=1,2, k=2 (2x2x1): rank 4 (schoolbook)
    - Block 2c: j=1,2, k=3,4 (2x2x2): rank 7 (Strassen)
    Total rank: 10 + 7 + 4 + 7 = 28
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 28
    V = [[0] * rank for _ in range(m * p)]   # 15 x 28
    W = [[0] * rank for _ in range(p * n)]   # 10 x 28
    
    col = 0
    
    # Block 1: j=0, k=0..4 (schoolbook for 2x1x5)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen for 2x2x2)
    # M1: (A[0,1]+A[1,2])(B[1,0]+B[2,1])
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2])(B[1,0])
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1
    W[0 * n + 1][col] = 1; W[1 * n + 1][col] = -1
    col += 1
    
    # M3: A[0,1](B[1,1]-B[2,1])
    U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1; V[2 * p + 1][col] = -1
    W[1 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M4: A[1,2](B[2,0]-B[1,0])
    U[1 * m + 2][col] = 1
    V[2 * p + 0][col] = 1; V[1 * p + 0][col] = -1
    W[0 * n + 0][col] = 1; W[0 * n + 1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2])(B[2,1])
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = -1; W[1 * n + 0][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1])(B[1,0]+B[1,1])
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2])(B[2,0]+B[2,1])
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    
    # Block 2b: j=1,2, k=2 (schoolbook for 2x2x1)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen for 2x2x2)
    # M1: (A[0,1]+A[1,2])(B[1,3]+B[2,4])
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2])(B[1,3])
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1
    W[3 * n + 1][col] = 1; W[4 * n + 1][col] = -1
    col += 1
    
    # M3: A[0,1](B[1,4]-B[2,4])
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1; V[2 * p + 4][col] = -1
    W[4 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M4: A[1,2](B[2,3]-B[1,3])
    U[1 * m + 2][col] = 1
    V[2 * p + 3][col] = 1; V[1 * p + 3][col] = -1
    W[3 * n + 0][col] = 1; W[3 * n + 1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2])(B[2,4])
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = -1; W[4 * n + 0][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1])(B[1,3]+B[1,4])
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 3][col] = 1; V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2])(B[2,3]+B[2,4])
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1
    col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    return {"U": U, "V": V, "W": W}
