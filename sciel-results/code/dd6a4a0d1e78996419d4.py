DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n  # 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(n):
        for j in range(m):
            U[r * m + j][r] = 1
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = 1
        for k in range(p):
            W[k * n + r][r] = 1
    
    return {"U": U, "V": V, "W": W}
