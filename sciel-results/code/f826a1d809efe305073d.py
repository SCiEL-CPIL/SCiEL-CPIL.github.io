DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 10 decomposition
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            c = i * p + k
            W[k * n + i][c] = 1
            for j in range(m):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
    
    return {"U": U, "V": V, "W": W}
