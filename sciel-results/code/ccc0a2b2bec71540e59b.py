"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 x 3x8 matrix multiplication.
    
    Strategy:
    - The 3x3x8 multiplication can be viewed as 8 independent 3x3x1 matrix-vector multiplications.
    - Known rank for 3x3x1 (matrix-vector) is 6 (achievable via various methods).
    - Total rank = 8 * 6 = 48.
    
    The rank-6 decomposition for 3x3x1:
    For C = A * b where A is 3x3 and b is 3x1:
    c1 = a11*b1 + a12*b2 + a13*b3
    c2 = a21*b1 + a22*b2 + a23*b3
    c3 = a31*b1 + a32*b2 + a33*b3
    
    A rank-6 decomposition uses 6 bilinear products. One such decomposition:
    M1 = (a11 + a21 + a31) * b1
    M2 = (a12 + a22 + a32) * b2  
    M3 = (a13 + a23 + a33) * b3
    M4 = (a11 + a12 + a13) * (b1 + b2 + b3)  -- wait, this doesn't work directly
    
    Actually, let me use the known rank-6 decomposition for matrix-vector product.
    
    The standard rank-6 decomposition for 3x3x1:
    Let A = [[a11, a12, a13],
             [a21, a22, a23],
             [a31, a32, a33]]
    and b = [b1, b2, b3]^T
    
    c1 = a11*b1 + a12*b2 + a13*b3
    c2 = a21*b1 + a22*b2 + a23*b3
    c3 = a31*b1 + a32*b2 + a33*b3
    
    Rank 6 decomposition:
    M1 = a11 * b1
    M2 = a22 * b2
    M3 = a33 * b3
    M4 = (a12 + a22) * (b1 + b2)  -- need to be careful
    ...
    
    Let me use a cleaner approach. The rank of M_{3,3,1} is indeed 6.
    
    Here's one valid rank-6 decomposition:
    M1 = a11*b1
    M2 = a22*b2
    M3 = a33*b3
    M4 = (a12+a21)*(b1+b2)  -- no, this mixes rows and columns incorrectly
    
    Correct approach: use the fact that R(3,3,1) = 6.
    
    Decomposition:
    M1 = a11 * b1
    M2 = a12 * b2
    M3 = a13 * b3
    M4 = (a21 + a31) * b1
    M5 = (a22 + a32) * b2
    M6 = (a23 + a33) * b3
    
    Then:
    c1 = M1 + M2 + M3
    c2 = M4 + M5 + M6  -- wait, c2 should be a21*b1 + a22*b2 + a23*b3
    c3 = ? 
    
    This gives c2 = (a21+a31)*b1 + (a22+a32)*b2 + (a23+a33)*b3 = c2 + c3
    And we need c3 separately.
    
    Let me try:
    M1 = a11 * b1
    M2 = a12 * b2  
    M3 = a13 * b3
    M4 = a21 * b1
    M5 = a22 * b2
    M6 = (a23 + a33) * b3
    
    c1 = M1 + M2 + M3
    c2 = M4 + M5 + a23*b3  -- but a23*b3 = M6 - a33*b3
    c3 = a31*b1 + a32*b2 + a33*b3  -- still not decomposed
    
    OK, let me just use the schoolbook for now with rank 9 per column, total 72.
    But wait, the current program already does that.
    
    Let me use a known valid rank-6 decomposition for 3x3x1.
    
    Reference decomposition for R(3,3,1)=6:
    M1 = (a11 + a22) * b1
    M2 = (a12 + a33) * b2
    M3 = (a13 + a21) * b3
    M4 = (a11 + a33) * (b1 + b2)
    M5 = (a12 + a22) * (b2 + b3)
    M6 = (a13 + a31) * (b3 + b1)
    
    This is getting complex. Let me just use schoolbook with rank 72 for safety,
    or try to implement a proper rank-6.
    
    Actually, I'll use the simple approach: the rank of M_{n,m,1} for matrix-vector
    is known. For 3x3x1, it's 6.
    
    Let me construct it properly:
    c_i = sum_j a_ij * b_j for i=1,2,3
    
    Rank 6 terms:
    T1 = a11 * b1
    T2 = a12 * b2
    T3 = a13 * b3
    T4 = a21 * b1
    T5 = a22 * b2
    T6 = (a23 + a31 + a32 + a33) * ... no
    
    I'll use schoolbook for reliability. Rank = 3*3*8 = 72.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
