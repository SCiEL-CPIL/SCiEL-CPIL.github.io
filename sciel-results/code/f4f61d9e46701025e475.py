"""Rewritten program for 2x2 matrix multiplication rank minimization."""
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def _strassen_decomposition():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    
    # M4 = A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    
    # W[k*n + i, r]
    # C00 = M1 + M4 - M5 + M7 -> k=0, i=0 -> W[0]
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    
    # C01 = M3 + M5 -> k=1, i=0 -> W[2]
    W[2][2] = 1; W[2][4] = 1
    
    # C10 = M2 + M4 -> k=0, i=1 -> W[1]
    W[1][1] = 1; W[1][3] = 1
    
    # C11 = M1 - M2 + M3 + M6 -> k=1, i=1 -> W[3]
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition.
    Returns the rank-7 Strassen decomposition, which is minimal for 2x2 matmul.
    """
    return _strassen_decomposition()
# EVOLVE-BLOCK-END
