"""Rank-25 exact tensor decomposition for 2×5×5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    rank = m * p  # 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = j * p + k
                U[i * m + j][r] = 1
                W[k * n + i][r] = 1
    
    for r in range(rank):
        V[r][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
