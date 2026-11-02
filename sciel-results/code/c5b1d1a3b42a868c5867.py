DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = m  # 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(m):  # c corresponds to j = c
        for i in range(n):
            U[i * m + c][c] = 1
        for k in range(p):
            V[c * p + k][c] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
