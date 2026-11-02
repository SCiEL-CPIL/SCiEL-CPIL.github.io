import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U (A):
    # 0: a11, 1: a12, 2: a13
    # 3: a21, 4: a22, 5: a23
    # 6: a31, 7: a32, 8: a33
    
    # Indices for V (B):
    # 0: b11, 1: b12, 2: b13
    # 3: b21, 4: b22, 5: b23
    # 6: b31, 7: b32, 8: b33
    
    # Indices for W (C):
    # W[k*n+i, r] -> C[i, k+1]
    # 0: c11 (k=0, i=0)
    # 1: c21 (k=0, i=1)
    # 2: c31 (k=0, i=2)
    # 3: c12 (k=1, i=0)
    # 4: c22 (k=1, i=1)
    # 5: c32 (k=1, i=2)
    # 6: c13 (k=2, i=0)
    # 7: c23 (k=2, i=1)
    # 8: c33 (k=2, i=2)
    
    col = 0
    
    # Strassen terms for A_TL * B_TL
    # M1 = (a11+a22)(b11+b22)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (a21+a22)b11
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = a11(b12-b22)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = a22(b21-b11)
    U[4][col] = 1
    V[0][col] = -1; V[3][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (a11+a12)b22
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (a21-a11)(b11+b12)
    U[0][col] = -1; U[3][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Cross terms
    # C13 = a11*b13 + a12*b23 + a13*b33
    # T8
    U[0][col] = 1
    V[2][col] = 1
    W[6][col] = 1
    col += 1
    
    # T9
    U[1][col] = 1
    V[5][col] = 1
    W[6][col] = 1
    col += 1
    
    # T10
    U[2][col] = 1
    V[8][col] = 1
    W[6][col] = 1
    col += 1
    
    # C23 = a21*b13 + a22*b23 + a23*b33
    # T11
    U[3][col] = 1
    V[2][col] = 1
    W[7][col] = 1
    col += 1
    
    # T12
    U[4][col] = 1
    V[5][col] = 1
    W[7][col] = 1
    col += 1
    
    # T13
    U[5][col] = 1
    V[8][col] = 1
    W[7][col] = 1
    col += 1
    
    # C31 = a31*b11 + a32*b21 + a33*b31
    # T14
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # T15
    U[7][col] = 1
    V[3][col] = 1
    W[2][col] = 1
    col += 1
    
    # T16
    U[8][col] = 1
    V[6][col] = 1
    W[2][col] = 1
    col += 1
    
    # C32 = a31*b12 + a32*b22 + a33*b32
    # T17
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # T18
    U[7][col] = 1
    V[4][col] = 1
    W[5][col] = 1
    col += 1
    
    # T19
    U[8][col] = 1
    V[7][col] = 1
    W[5][col] = 1
    col += 1
    
    # C33 = a31*b13 + a32*b23 + a33*b33
    # T20
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    # T21
    U[7][col] = 1
    V[5][col] = 1
    W[8][col] = 1
    col += 1
    
    # T22
    U[8][col] = 1
    V[8][col] = 1
    W[8][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
