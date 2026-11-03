def construct():
    n, m, p = DIMENSIONS
    rank = n  # 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        c = i
        for j in range(m):
            U[i * m + j][c] = 1
            V[j * p + i][c] = 1
        W[i * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
