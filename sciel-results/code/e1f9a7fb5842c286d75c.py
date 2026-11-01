import time
import numpy as np

DIMENSIONS = [2, 3, 10]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 2.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-55 exact tensor decomposition for 2x3 @ 3x10.
    
    Strategy:
    Split the middle dimension m=3 into m1=2 and m2=1.
    A = [A1 (2x2), A2 (2x1)]
    B = [B1 (2x10); B2 (1x10)]
    C = A1 B1 + A2 B2
    
    Part 1: A1 B1 (2x2 @ 2x10)
    Split the last dimension p=10 into 5 pairs of 2.
    Each 2x2 @ 2x2 block uses Strassen's algorithm (rank 7).
    Total rank for Part 1: 5 * 7 = 35.
    
    Part 2: A2 B2 (2x1 @ 1x10)
    Each 2x1 @ 1x1 block uses schoolbook (rank 2).
    Total rank for Part 2: 10 * 2 = 20.
    
    Total rank: 55.
    """
    n, m, p = DIMENSIONS
    rank = 55
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) @ B1 (2x10)
    # A1 corresponds to A[0:2, 0:2]
    # B1 corresponds to B[0:2, :]
    # C1 corresponds to C[:, :]
    
    for k_start in range(0, p, 2):
        k1 = k_start
        k2 = k_start + 1
        
        # Strassen terms for A[0:2, 0:2] @ B[0:2, k1:k2+1]
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k2][col] = 1.0
        W[k1*n + 0][col] = 1.0; W[k2*n + 1][col] = 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        
        # M3 = A00(B10-B11)
        U[0][col] = 1.0
        V[1*p + k1][col] = 1.0; V[1*p + k2][col] = -1.0
        W[k1*n + 0][col] = 1.0
        col += 1
        
        # M4 = A11(B01-B00)
        U[3][col] = 1.0
        V[0*p + k2][col] = 1.0; V[0*p + k1][col] = -1.0
        W[k1*n + 1][col] = 1.0; W[k2*n + 0][col] = 1.0
        col += 1
        
        # M5 = (A00+A10)B11
        U[0][col] = 1.0; U[2][col] = 1.0
        V[1*p + k2][col] = 1.0
        W[k2*n + 0][col] = -1.0
        col += 1
        
        # M6 = (A11-A00)(B00+B10)
        U[3][col] = 1.0; U[0][col] = -1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k2*n + 1][col] = 1.0
        col += 1
        
        # M7 = (A01-A11)(B11+B01)
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k2][col] = 1.0; V[0*p + k2][col] = 1.0
        W[k2*n + 0][col] = 1.0
        col += 1

    # Part 2: A2 (2x1) @ B2 (1x10)
    # A2 corresponds to A[0:2, 2]
    # B2 corresponds to B[2, :]
    # C2 corresponds to C[:, :]
    
    for k in range(p):
        # Schoolbook for 2x1 @ 1x1
        # C0 = A0 B
        U[0][col] = 1.0
        V[2*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        
        # C1 = A1 B
        U[1][col] = 1.0
        V[2*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
