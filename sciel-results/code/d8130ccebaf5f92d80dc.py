import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m  # 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            for k in range(p):
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
