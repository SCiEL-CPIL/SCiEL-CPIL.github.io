"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-7 tensor decomposition for 2x2 matrix multiplication.
    
    This uses Strassen's algorithm, which is provably optimal for 2x2 matrices.
    
    Interface:
    - U: shape (n*m, rank) = (4, 7), row index = i*m + j
    - V: shape (m*p, rank) = (4, 7), row index = j*p + k  
    - W: shape (p*n, rank) = (4, 7), row index = k*n + i
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    This represents the bilinear map C[i,k] = sum_j A[i,j] * B[j,k].
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 7
    V = [[0] * rank for _ in range(m * p)]  # 4 x 7
    W = [[0] * rank for _ in range(p * n)]  # 4 x 7
    
    # Strassen decomposition for 2x2 matrix multiplication:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    #
    # Products:
    # r=0: M1 = (a+d)(e+h)
    # r=1: M2 = (c+d)e
    # r=2: M3 = a(f-h)
    # r=3: M4 = d(g-e)
    # r=4: M5 = (a+b)h
    # r=5: M6 = (c-a)(e+f)
    # r=6: M7 = (b-d)(g+h)
    #
    # Reconstruction:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Index mapping:
    # U rows: 0(a), 1(b), 2(c), 3(d)
    # V rows: 0(e), 1(f), 2(g), 3(h)
    # W rows: 0(C00), 1(C10), 2(C01), 3(C11)
    
    # r=0: M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # r=2: M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # r=3: M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # r=4: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # r=5: M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # r=6: M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
