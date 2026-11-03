def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: (i, j)
                U[i * m + j][c] = 1
                # V index: (j, k)
                V[j * p + k][c] = 1
                # W index: (k, i)
                W[k * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
