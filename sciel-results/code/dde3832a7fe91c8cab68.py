def construct():
    n, m, p = DIMENSIONS
    rank = n * min(m, p)  # 2 * 3 = 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            if j < p:  # j==k is only possible when j < p
                U[i * m + j][col] = 1
                V[j * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
