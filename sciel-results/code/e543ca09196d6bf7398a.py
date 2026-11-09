"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    n, m, p = 2, 5, 6
    
    # Trivial decomposition: rank = n*m*p = 60
    # For each (i,j,k), one rank-1 term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    # Deadline guard
    if time.time() - start_time > 100:
        pass  # Already done, just return
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
