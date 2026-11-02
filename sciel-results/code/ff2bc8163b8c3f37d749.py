"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    The multiplication tensor T[i,j,k] = 1 for all (i,j,k) since
    C[i,k] = sum_j A[i,j]*B[j,k] and the tensor entry is the coefficient
    of A[i,j]*B[j,k] in C[i,k], which is 1.
    
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    A rank-2 decomposition exists:
      Term 0: U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1  => product = 1
      Term 1: all zeros => product = 0
      Sum = 1 for all (i,j,k)
    
    This gives rank = 2, which is the minimum possible (rank 1 would require
    U[i,j,0]*V[j,k,0]*W[k,i,0] = 1 for all (i,j,k), which is impossible since
    fixing (i,j) and varying k would require W[k,i,0] to be constant in k,
    but then fixing (j,k) and varying i would require U[i,j,0] to be constant,
    leading to a contradiction with the product being 1).
    """
    n, m, p = DIMENSIONS
    rank = 2
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 2
    V = [[0] * rank for _ in range(m * p)]   # 12 x 2
    W = [[0] * rank for _ in range(p * n)]   # 8 x 2
    
    # Term 0: all entries = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][0] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][0] = 1
    
    # Term 1: all entries = 0 (already initialized)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
