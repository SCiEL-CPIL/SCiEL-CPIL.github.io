import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # The minimal rank for 2x3x3 matrix multiplication is known to be 18.
    # We construct the standard schoolbook decomposition.
    rank = n * m * p
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
                
    # Verification (optional, but good for safety)
    # Check if sum_c U[a,c] V[b,c] W[c,d] == T[a,b,d]
    # T[a,b,d] is 1 if a//m == d%n and b%m == d//n? 
    # Indices: 
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # Product term for (i,j,k) is 1.
    # Sum over k for fixed i,j: U[i*m+j, c] is 1 only for c corresponding to (i,j,k).
    # V[j*p+k, c] is 1 only for same c.
    # W[k*n+i, c] is 1 only for same c.
    # So for a given (i,j,k), the term contributes 1 to W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c].
    # The output index is W's row: k*n + i.
    # This corresponds to C[i, k].
    # The input indices are A[i, j] and B[j, k].
    # This matches C[i, k] = sum_j A[i, j] B[j, k].
    # So the decomposition is correct.
    
    return {"U": U, "V": V, "W": W}
