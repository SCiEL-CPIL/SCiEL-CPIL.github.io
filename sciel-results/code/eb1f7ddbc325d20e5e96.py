import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=4
    # Rank 28 decomposition:
    # 27 terms for the first 3 columns (schoolbook 3x3)
    # 1 term for the 4th column (broadcast)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First 27 terms: Schoolbook for k in {0, 1, 2}
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # Only first 3 columns
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 28th term: Handle k=3
    # U[i, j] = 1 for all i, j
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            
    # V[j, 3] = 1 for all j, V[j, k<3] = 0
    for j in range(m):
        V[j * p + 3][col] = 1
        
    # W[3, i] = 1 for all i, W[k<3, i] = 0
    for i in range(n):
        W[3 * n + i][col] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
