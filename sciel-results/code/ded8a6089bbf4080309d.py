"""Matrix multiplication 3x4x6 tensor decomposition."""
DIMENSIONS = [3, 4, 6]
import time

def construct():
    """
    Return a rank-minimal (or at least improved) exact tensor decomposition
    for multiplying a 3x4 matrix by a 4x6 matrix.
    
    Uses block decomposition:
    - Split A (3x4) into A1 (3x3) and A2 (3x1)
    - Split B (4x6) into B1 (3x6) and B2 (1x6)
    - C = A1*B1 + A2*B2
    
    For A1*B1 (3x3 times 3x6), split B1 into B1a (3x3) and B1b (3x3):
    - A1*B1a: 3x3x3 schoolbook, rank 27
    - A1*B1b: 3x3x3 schoolbook, rank 27
    
    For A2*B2 (3x1 times 1x6): outer product, rank 6
    
    Total rank: 27 + 27 + 6 = 60
    """
    n, m, p = 3, 4, 6  # A is n x m, B is m x p, C is n x p
    
    # Total rank
    rank = 60
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]  # 12 x 60
    V = [[0] * rank for _ in range(m * p)]  # 24 x 60
    W = [[0] * rank for _ in range(p * n)]  # 18 x 60
    
    col = 0
    
    # Block 1: A1 (rows 0-2 of A, cols 0-2) times B1a (rows 0-2 of B, cols 0-2 of B)
    # This computes C[0:3, 0:3] = A[0:3, 0:3] * B[0:3, 0:3]
    # Schoolbook for 3x3x3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Block 2: A1 (rows 0-2 of A, cols 0-2) times B1b (rows 0-2 of B, cols 3-5 of B)
    # This computes C[0:3, 3:6] = A[0:3, 0:3] * B[0:3, 3:6]
    # Schoolbook for 3x3x3
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + (k + 3)][col] = 1
                W[(k + 3) * n + i][col] = 1
                col += 1
    
    # Block 3: A2 (rows 0-2 of A, col 3) times B2 (row 3 of B, cols 0-5 of B)
    # This computes C[0:3, 0:6] += A[0:3, 3] * B[3, 0:6]
    # Outer product: rank 6
    for i in range(3):
        for k in range(6):
            U[i * m + 3][col] = 1
            V[3 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
