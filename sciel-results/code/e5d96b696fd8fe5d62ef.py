def construct():
    n, m, p = 2, 3, 9
    rank = 3
    U = [[1] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            q = k // 3
            V[j * p + k][q] = 1
    
    for k in range(p):
        q = k // 3
        for i in range(n):
            W[k * n + i][q] = 1
    
    return {"U": U, "V": V, "W": W}
