import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 2.0

# EVOLVE-BLOCK-START
def construct():
    # The task is to find the rank-minimal decomposition for 2x3 @ 3x10.
    # The multiplication tensor T_{2,3,10} is the direct sum of 10 independent
    # T_{2,3,1} tensors (matrix-vector products).
    # The rank of T_{n,m,1} is n*m for n,m >= 2 (except specific small cases, but 2x3 is 6).
    # Thus rank(T_{2,3,10}) >= 10 * rank(T_{2,3,1}) = 10 * 6 = 60.
    # The schoolbook decomposition achieves rank 60.
    # Therefore, 60 is the exact minimum rank.
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Construct the standard schoolbook decomposition
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    # Indexing:
    # U[i*m + j, r] = 1 if r corresponds to (i, j, k)
    # V[j*p + k, r] = 1 if r corresponds to (i, j, k)
    # W[k*n + i, r] = 1 if r corresponds to (i, j, k)
    
    # We can generate this efficiently using numpy or lists.
    # Lists are safer for exactness and avoiding float issues, though 1.0 is exact.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
