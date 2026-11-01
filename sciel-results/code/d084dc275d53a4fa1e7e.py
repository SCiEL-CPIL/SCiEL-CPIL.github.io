def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 2 * 16 = 32
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 32)
    V = [[0] * rank for _ in range(m * p)]  # (32, 32)
    W = [[0] * rank for _ in range(p * n)]  # (32, 32)
    
    for k in range(p):
        for s in range(m):  # s = 0, 1
            r = k * m + s
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = 1 if j == s else 0
                V[s * p + k][r] = 1  # j' = s
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
