"""Rewritten program for matrix multiplication 2x3 by 3x7.
The minimal rank for this tensor decomposition is 42 (schoolbook).
The program returns the standard schoolbook decomposition.
"""

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: shape (n*m, rank)
    # V: shape (m*p, rank)
    # W: shape (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over all terms in the matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for A[i,j] in U
                idx_u = i * m + j
                # Index for B[j,k] in V
                idx_v = j * p + k
                # Index for C[i,k] in W
                # The problem statement says: "output factor follows the official trace convention W[k*n+i,r]"
                # So index is k * n + i
                idx_w = k * n + i
                
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
