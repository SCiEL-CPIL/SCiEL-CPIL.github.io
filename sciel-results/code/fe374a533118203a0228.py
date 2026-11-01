import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    NM, MP, PN = n * m, m * p, p * n
    
    U = [[0.0] * rank for _ in range(NM)]
    V = [[0.0] * rank for _ in range(MP)]
    W = [[0.0] * rank for _ in range(PN)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
