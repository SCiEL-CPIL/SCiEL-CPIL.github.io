"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook rank-72 decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 72
    
    U = [[0] * rank for _ in range(n * m)]  # (9, 72)
    V = [[0] * rank for _ in range(m * p)]  # (24, 72)
    W = [[0] * rank for _ in range(p * n)]  # (24, 72)
    
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
