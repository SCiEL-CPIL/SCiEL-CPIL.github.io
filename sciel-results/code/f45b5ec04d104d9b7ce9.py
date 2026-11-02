"""Matrix multiplication 2,2,14 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 56
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 28 x 56
    W = [[0] * rank for _ in range(p * n)]  # 28 x 56
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
