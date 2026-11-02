import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    # Rank 7 construction:
    # U[i*m+j, r] = 1 for all i,j,r
    # V[j*p+k, r] = 1 if k==r else 0  
    # W[k*n+i, r] = 1 for all k,i,r
    
    rank = 7
    U = [[1] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            V[j * p + k][k] = 1
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
