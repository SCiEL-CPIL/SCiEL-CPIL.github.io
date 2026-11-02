"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-60 schoolbook decomposition for 3x4x5 matrix multiplication.
    
    The tensor T_{i,j,k} = A[i,j] * B[j,k] is decomposed as:
    T = sum_{r=1}^{60} U[:,r] V[:,r]^T W[:,r]^T
    
    where for each (i,j,k) triple:
    U[i*m+j, r] = 1
    V[j*p+k, r] = 1  
    W[k*n+i, r] = 1
    
    This is the minimal rank decomposition known for the general case.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    rank = n * m * p  # 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
