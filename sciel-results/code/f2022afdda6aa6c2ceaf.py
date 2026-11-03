"""Matrix multiplication 3x4 * 4x5 with rank-minimal exact tensor decomposition."""

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook decomposition: rank = n*m*p = 60
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 12 x 60
    V = [[0] * rank for _ in range(m * p)]  # 20 x 60
    W = [[0] * rank for _ in range(p * n)]  # 15 x 60
    
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
