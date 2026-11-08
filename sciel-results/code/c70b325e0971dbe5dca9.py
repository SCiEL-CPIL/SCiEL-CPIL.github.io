<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Decompose into n independent 1 x m x p multiplications.
    # Rank of 1 x m x p is max(m, p).
    r_block = max(m, p)
    rank = n * r_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for r in range(r_block):
            col = i * r_block + r
            # U[i*m + j, col] = 1 if j == r
            if r < m:
                U[i * m + r][col] = 1
            
            # W[k*n + i, col] = 1 if k == r
            if r < p:
                W[r * n + i][col] = 1
            
            # V[j*p + k, col] = 1 if j == r and k == r
            if r < m and r < p:
                V[r * p + r][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
