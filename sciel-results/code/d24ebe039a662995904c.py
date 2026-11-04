"""Program for 2x3x6 matrix multiplication tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition:
    # Split A (2x3) into A1 (2x1) and A2 (2x2)
    # Split B (3x6) into B1 (1x6) and B2 (2x6)
    # C = A1*B1 + A2*B2
    # A1*B1: 2x1x6, rank 12 (schoolbook)
    # A2*B2: 2x2x6, split into three 2x2x2 blocks, each rank 7 (Strassen), total 21
    # Total rank: 33
    
    n, m, p = 2, 3, 6
    rank = 33
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x6) => 2x6 output
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Part 2: A2 (2x2) * B2 (2x6) => 2x6 output
    # Split B2 into three 2x2 blocks, use Strassen for each
    
    for t in range(3):
        # M1 = (A00+A11)(B00+B11)
        U[0 * m + 1, col] = 1.0
        U[1 * m + 2, col] = 1.0
        V[1 * p + 2*t, col] = 1.0
        V[2 * p + 2*t + 1, col] = 1.0
        W[2*t * n + 0, col] = 1.0
        W[(2*t + 1) * n + 1, col] = 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[1 * m + 1, col] = 1.0
        U[1 * m + 2, col] = 1.0
        V[1 * p + 2*t, col] = 1.0
        W[2*t * n + 1, col] = 1.0
        W[(2*t + 1) * n + 1, col] = -1.0
        col += 1
        
        # M3 = A00(B01-B11)
        U[0 * m + 1, col] = 1.0
        V[1 * p + 2*t + 1, col] = 1.0
        V[2 * p + 2*t + 1, col] = -1.0
        W[(2*t + 1) * n + 0, col] = 1.0
        W[(2*t + 1) * n + 1, col] = 1.0
        col += 1
        
        # M4 = A11(B10-B00)
        U[1 * m + 2, col] = 1.0
        V[2 * p + 2*t, col] = 1.0
        V[1 * p + 2*t, col] = -1.0
        W[2*t * n + 0, col] = 1.0
        W[2*t * n + 1, col] = 1.0
        col += 1
        
        # M5 = (A00+A01)B11
        U[0 * m + 1, col] = 1.0
        U[0 * m + 2, col] = 1.0
        V[2 * p + 2*t + 1, col] = 1.0
        W[2*t * n + 0, col] = -1.0
        W[(2*t + 1) * n + 0, col] = 1.0
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1 * m + 1, col] = 1.0
        U[0 * m + 1, col] = -1.0
        V[1 * p + 2*t, col] = 1.0
        V[1 * p + 2*t + 1, col] = 1.0
        W[(2*t + 1) * n + 1, col] = 1.0
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0 * m + 2, col] = 1.0
        U[1 * m + 2, col] = -1.0
        V[2 * p + 2*t, col] = 1.0
        V[2 * p + 2*t + 1, col] = 1.0
        W[2*t * n + 0, col] = 1.0
        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
