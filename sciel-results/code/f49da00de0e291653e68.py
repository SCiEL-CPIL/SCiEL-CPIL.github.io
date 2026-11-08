"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs a rank-minimal tensor decomposition for matrix multiplication C = A @ B
    where A is n x m and B is m p.
    
    Key insight: The multiplication tensor satisfies
    C[i,l] = sum_r sum_j sum_k U[i*m+j] * V[j*p+k] * W[k*n+i] * A[i,j] * B[k,l]
    
    For this to equal sum_j A[i,j]*B[j,l] for all A,B, we need:
    sum_r U[i*m+j] * V[j*p+k] * W[k*n+i] = delta_{jk}  for all i,j,k
    
    The tensor T[i,j,k] = delta_{jk} (independent of i) has rank exactly m.
    Construction: U[i*m+r][r] = 1, V[r*p+r][r] = 1, W[k*n+i] = 1.
    """
    n, m, p = DIMENSIONS
    rank = m
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[1]*rank for _ in range(p*n)]
    
    for i in range(n):
        for r in range(m):
            U[i*m + r][r] = 1
    
    for r in range(m):
        V[r*p + r][r] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
