"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time

def construct():
    # Schoolbook decomposition of rank n*m*p
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    rank = n * m * p  # 52
    U = [[0] * rank for _ in range(n * m)]  # 4 x 52
    V = [[0] * rank for _ in range(m * p)]  # 26 x 52
    W = [[0] * rank for _ in range(p * n)]  # 26 x 52
    
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
