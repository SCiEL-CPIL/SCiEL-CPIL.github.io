"""Program that constructs a tensor decomposition for 3x5x7 matrix multiplication
using a schoolbook decomposition with a time guard.
"""

import time

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    
    # Schoolbook decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
    # Rank = n*m*p = 105
    # U indexed by (i,j) -> i*m + j, shape (n*m, rank)
    # V indexed by (j,k) -> j*p + k, shape (m*p, rank)  
    # W indexed by (k,i) -> k*n + i, shape (p*n, rank)
    # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
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
    
    # Time guard: return immediately if approaching the limit
    elapsed = time.time() - start_time
    if elapsed > 140:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
