"""Matrix multiplication 2x2 by 2x10: rank-35 Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Uses Strassen's algorithm on 5 independent 2x2x2 blocks.
    Rank = 5 * 7 = 35.
    
    Interface:
    U: shape (n*m, rank) = (4, 35)
    V: shape (m*p, rank) = (20, 35)
    W: shape (p*n, rank) = (20, 35)
    
    Indices (official trace convention W[k*n+i,r]):
    U index: i*m + j
    V index: j*p + k  
    W index: k*n + i
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen's 7 products for M1 (2x2) * M2 (2x2):
    # M1 = [[a, b], [c, d]], M2 = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)*e
    # P3 = a*(f-h)
    # P4 = d*(g-e)
    # P5 = (a+b)*h
    # P6 = (c-a)*(e+f)
    # P7 = (b-d)*(g+h)
    #
    # Result:
    # M[0,0] = P1 + P4 - P5 + P7
    # M[0,1] = P3 + P5
    # M[1,0] = P2 + P4
    # M[1,1] = P1 - P2 + P3 + P6
    
    strassen_A = [
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a+b
        [-1, 0, 1, 0],  # P6: c-a
        [0, 1, 0, -1]   # P7: b-d
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # P1: e+h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f-h
        [-1, 0, 1, 0],  # P4: g-e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e+f
        [0, 0, 1, 1]    # P7: g+h
    ]
    
    # W contributions: [C00, C01, C10, C11]
    w_contrib = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, -1],  # P2
        [0, 1, 0, 1],   # P3
        [1, 0, 1, 0],   # P4
        [-1, 1, 0, 0],  # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0]    # P7
    ]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # U: index i*m + j
            U[0][col] = strassen_A[r][0]  # a
            U[1][col] = strassen_A[r][1]  # b
            U[2][col] = strassen_A[r][2]  # c
            U[3][col] = strassen_A[r][3]  # d
            
            # V: index j*p + k
            k0 = 2 * b
            k1 = 2 * b + 1
            V[0 * p + k0][col] = strassen_B[r][0]  # e
            V[0 * p + k1][col] = strassen_B[r][1]  # f
            V[1 * p + k0][col] = strassen_B[r][2]  # g
            V[1 * p + k1][col] = strassen_B[r][3]  # h
            
            # W: index k*n + i
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]      # C00
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]  # C01
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]      # C10
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]  # C11
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
