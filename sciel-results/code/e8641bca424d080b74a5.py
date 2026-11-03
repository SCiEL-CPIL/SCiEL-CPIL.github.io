"""
Optimized construction for 4x5x5 matrix multiplication tensor decomposition.
Uses the standard schoolbook decomposition which has rank n*m*p = 100.
This is the known trivial upper bound. No better exact decomposition is 
immediately available in the standard library or simple heuristics within 
the time budget.
"""

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices for the schoolbook decomposition
    # The tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each (i, j, k), we assign a unique column c = i*m*p + j*p + k
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
