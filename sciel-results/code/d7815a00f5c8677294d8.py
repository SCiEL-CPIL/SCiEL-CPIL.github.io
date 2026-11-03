"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time

# EVOLVE-BLOCK-START
def construct():
    """Schoolbook decomposition of matrix multiplication 5x5x5 with rank 125."""
    start = time.time()
    deadline = start + 100  # safety margin before 120s kill
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    rank = n * m * p  # 125
    
    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U[i*m + j, c] = 1 for term c corresponding to (i,j,k)
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # where c = i*m*p + j*p + k
    
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
