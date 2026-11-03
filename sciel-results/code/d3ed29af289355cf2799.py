"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Structured decomposition of rank n*m
    # T[i,j,k] = delta_{i,k}
    # For each i in 0..n-1, j in 0..m-1, use one term
    # U[i*m+j, c] = 1, V[j*p+i, c] = 1, W[i*n+i, c] = 1 where c = i*m+j
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
            V[j * p + i][c] = 1
            W[i * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
