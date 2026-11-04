"""Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Build a rank-33 exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy:
    - Split A (2x3) into A1 (2x1, col 0) and A2 (2x2, cols 1-2)
    - Split B (3x6) into B1 (1x6, row 0) and B2 (2x6, rows 1-2)
    - C = A1*B1 + A2*B2
    - A1*B1: schoolbook, rank 12
    - A2*B2: split into three 2x2*2x2 blocks, each Strassen rank 7, total 21
    - Total rank: 33
    """
    n, m, p = 2, 3, 6
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x6) => 2x6 output
    # A1 has elements A[0][0], A[1][0]
    # B1 has elements B[0][k] for k=0..5
    # C[i][k] += A[i][0] * B[0][k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 12
    
    # Part 2: A2 (2x2) * B2 (2x6) => 2x6 output
    # A2 elements: A[0][1], A[0][2], A[1][1], A[1][2]
    # B2 elements: B[1][k], B[2][k] for k=0..5
    # Split B2 into three 2x2 blocks: cols (0,1), (2,3), (4,5)
    
    for t in range(3):
        # Block t: B2 cols are 2t, 2t+1
        # Strassen for 2x2*2x2:
        # A00=A[0][1], A01=A[0][2], A10=A[1][1], A11=A[1][2]
        # B00=B[1][2t], B01=B[1][2t+1], B10=B[2][2t], B11=B[2][2t+1]
        # C00=C[0][2t], C01=C[0][2t+1], C10=C[1][2t], C11=C[1][2t+1]
        
        # M1 = (A00+A11)(B00+B11)
        U[0 * m + 1][col] = 1  # A[0][1]
        U[1 * m + 2][col] = 1  # A[1][2]
        V[1 * p + 2*t][col] = 1  # B[1][2t]
        V[2 * p + 2*t + 1][col] = 1  # B[2][2t+1]
        W[2*t * n + 0][col] = 1  # C[0][2t]
        W[(2*t + 1) * n + 1][col] = 1  # C[1][2t+1]
        col += 1
        
        # M2 = (A10+A11)B00
        U[1 * m + 1][col] = 1  # A[1][1]
        U[1 * m + 2][col] = 1  # A[1][2]
        V[1 * p + 2*t][col] = 1  # B[1][2t]
        W[2*t * n + 1][col] = 1  # C[1][2t]
        W[(2*t + 1) * n + 1][col] = -1  # C[1][2t+1]
        col += 1
        
        # M3 = A00(B01-B11)
        U[0 * m + 1][col] = 1  # A[0][1]
        V[1 * p + 2*t + 1][col] = 1  # B[1][2t+1]
        V[2 * p + 2*t + 1][col] = -1  # B[2][2t+1]
        W[(2*t + 1) * n + 0][col] = 1  # C[0][2t+1]
        W[(2*t + 1) * n + 1][col] = 1  # C[1][2t+1]
        col += 1
        
        # M4 = A11(B10-B00)
        U[1 * m + 2][col] = 1  # A[1][2]
        V[2 * p + 2*t][col] = 1  # B[2][2t]
        V[1 * p + 2*t][col] = -1  # B[1][2t]
        W[2*t * n + 0][col] = 1  # C[0][2t]
        W[2*t * n + 1][col] = 1  # C[1][2t]
        col += 1
        
        # M5 = (A00+A01)B11
        U[0 * m + 1][col] = 1  # A[0][1]
        U[0 * m + 2][col] = 1  # A[0][2]
        V[2 * p + 2*t + 1][col] = 1  # B[2][2t+1]
        W[2*t * n + 0][col] = -1  # C[0][2t]
        W[(2*t + 1) * n + 0][col] = 1  # C[0][2t+1]
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1 * m + 1][col] = 1  # A[1][1]
        U[0 * m + 1][col] = -1  # A[0][1]
        V[1 * p + 2*t][col] = 1  # B[1][2t]
        V[1 * p + 2*t + 1][col] = 1  # B[1][2t+1]
        W[(2*t + 1) * n + 1][col] = 1  # C[1][2t+1]
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0 * m + 2][col] = 1  # A[0][2]
        U[1 * m + 2][col] = -1  # A[1][2]
        V[2 * p + 2*t][col] = 1  # B[2][2t]
        V[2 * p + 2*t + 1][col] = 1  # B[2][2t+1]
        W[2*t * n + 0][col] = 1  # C[0][2t]
        col += 1
    
    # col should be 12 + 3*7 = 33
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
