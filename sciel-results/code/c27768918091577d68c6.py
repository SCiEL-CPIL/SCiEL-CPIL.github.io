"""
Optimized program for 2x2 matrix multiplication rank minimization.

The tensor rank of 2x2 matrix multiplication is exactly 7 (Strassen, 1969).
This program returns the canonical Strassen decomposition.
"""
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    
    Rank: 7
    Convention:
      U[i*m + j, r] : Coefficient of A[i,j] in the r-th product
      V[j*p + k, r] : Coefficient of B[j,k] in the r-th product
      W[k*n + i, r] : Coefficient of C[i,k] from the r-th product (trace convention: k*n + i)
      
    The decomposition satisfies:
      T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid i,j,k
      which corresponds to C[i,k] = sum_j A[i,j] B[j,k]
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 intermediate products:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indexing:
    # U indices: a=0, b=1, c=2, d=3
    # V indices: e=0, f=1, g=2, h=3
    # W indices: C00=0 (k=0,i=0), C10=1 (k=0,i=1), C01=2 (k=1,i=0), C11=3 (k=1,i=1)
    # Note: W index is k*n + i.
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[1,0] -> k=0, i=1 -> W[1]
    # C[0,1] -> k=1, i=0 -> W[2]
    # C[1,1] -> k=1, i=1 -> W[3]
    
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
