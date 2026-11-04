"""Matrix multiplication 3x3 x 3x6 tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    # Schoolbook decomposition: rank = n*m*p = 54
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 9 x 54
    V = [[0] * rank for _ in range(m * p)]  # 18 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54
    
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
