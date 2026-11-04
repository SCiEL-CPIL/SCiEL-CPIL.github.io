"""Matrix multiplication tensor decomposition for 3x3x4."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # Returns the schoolbook decomposition which is always valid.
    # Rank is n*m*p.
    n, m, p = DIMENSIONS
    r = n * m * p
    
    # Pre-allocate matrices with zeros
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Fill in the decomposition terms
    # Term (i,j,k) corresponds to column c = i*m*p + j*p + k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
