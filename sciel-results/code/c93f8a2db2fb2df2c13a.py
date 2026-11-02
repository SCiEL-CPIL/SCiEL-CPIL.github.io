"""
Rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
Uses Strassen's algorithm on 5 disjoint column-pairs of B.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    The tensor for C = A*B where A is 2x2 and B is 2x10 is decomposed as
    5 independent 2x2x2 multiplication blocks (one per pair of columns in B).
    Each block uses Strassen's rank-7 algorithm, giving total rank 5*7 = 35.
    
    Interface:
      U: shape (n*m, rank) = (4, 35)
      V: shape (m*p, rank) = (20, 35)
      W: shape (p*n, rank) = (20, 35)
    
    Index conventions:
      U index: i*m + j  (i in 0..1, j in 0..1) -> 0,1,2,3
      V index: j*p + k  (j in 0..1, k in 0..9) -> 0..19
      W index: k*n + i  (k in 0..9, i in 0..1) -> 0..19  (official trace convention)
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 35
    
    # Build U, V, W as lists of lists for exact integer arithmetic
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = [[a, b], [c, d]], M2 = [[e, f], [g, h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    #
    # A-indices: 0:a, 1:b, 2:c, 3:d
    # B-indices: 0:e, 1:f, 2:g, 3:h
    strassen_A = [
        [1,  0, 0,  1],   # P1: a+d
        [0,  0, 1,  1],   # P2: c+d
        [1,  0, 0,  0],   # P3: a
        [0,  0, 0,  1],   # P4: d
        [1,  1, 0,  0],   # P5: a+b
        [-1, 0, 1,  0],   # P6: c-a
        [0,  1, 0, -1],   # P7: b-d
    ]
    strassen_B = [
        [1,  0, 0,  1],   # P1: e+h
        [1,  0, 0,  0],   # P2: e
        [0,  1, 0, -1],   # P3: f-h
        [-1, 0, 1,  0],   # P4: g-e
        [0,  0, 0,  1],   # P5: h
        [1,  1, 0,  0],   # P6: e+f
        [0,  0, 1,  1],   # P7: g+h
    ]
    
    # W contributions: each Strassen product contributes to local C entries.
    # Local C layout: C00, C01, C10, C11
    # M00 = P1 + P4 - P5 + P7
    # M01 = P3 + P5
    # M10 = P2 + P4
    # M11 = P1 - P2 + P3 + P6
    w_contrib = [
        [1,  0,  0,  1],   # P1:  C00=+1, C11=+1
        [0,  0,  1, -1],   # P2:  C10=+1, C11=-1
        [0,  1,  0,  1],   # P3:  C01=+1, C11=+1
        [1,  0,  1,  0],   # P4:  C00=+1, C10=+1
        [-1, 1,  0,  0],   # P5:  C00=-1, C01=+1
        [0,  0,  0,  1],   # P6:  C11=+1
        [1,  0,  0,  0],   # P7:  C00=+1
    ]
    
    col = 0
    for b in range(5):
        if time.time() > deadline:
            break
        for r in range(7):
            # U: same A-coefficients for every block
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            # V: B-coefficients mapped to global indices for block b
            # e = B[0, 2b]   -> index 0*p + 2b     = 2b
            # f = B[0, 2b+1] -> index 0*p + 2b+1   = 2b+1
            # g = B[1, 2b]   -> index 1*p + 2b     = 10 + 2b
            # h = B[1, 2b+1] -> index 1*p + 2b+1   = 10 + 2b+1
            V[2*b][col]       = strassen_B[r][0]
            V[2*b+1][col]     = strassen_B[r][1]
            V[10 + 2*b][col]  = strassen_B[r][2]
            V[10 + 2*b+1][col]= strassen_B[r][3]
            
            # W: C-coefficients mapped to global indices (trace convention k*n+i)
            # C00: i=0, k=2b     -> 2b*2 + 0 = 4b
            # C01: i=0, k=2b+1   -> (2b+1)*2 + 0 = 4b+2
            # C10: i=1, k=2b     -> 2b*2 + 1 = 4b+1
            # C11: i=1, k=2b+1   -> (2b+1)*2 + 1 = 4b+3
            W[4*b][col]    = w_contrib[r][0]   # C00
            W[4*b+2][col]  = w_contrib[r][1]   # C01
            W[4*b+1][col]  = w_contrib[r][2]   # C10
            W[4*b+3][col]  = w_contrib[r][3]   # C11
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
