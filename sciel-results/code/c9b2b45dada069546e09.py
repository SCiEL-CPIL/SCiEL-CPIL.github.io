"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # For matrix multiplication (n,m,p), the tensor T[i,j,k] = delta[j,k]
    # We can decompose this with rank = m (one term for each j=k pair)
    # For (2,3,7), rank = 3
    
    rank = m  # 3
    
    # U: shape (n*m, rank) = (6, 3)
    # U[i*m + j, r] = 1 for all i, j, r (all ones)
    U = [[1] * rank for _ in range(n * m)]
    
    # V: shape (m*p, rank) = (21, 3)
    # V[j*p + k, r] = 1 if j == r else 0
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    # W: shape (p*n, rank) = (14, 3)
    # W[k*n + i, r] = 1 if k == r else 0
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            W[k * n + i][k] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
