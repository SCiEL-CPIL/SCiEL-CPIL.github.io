DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Part 1: rows 0,1 (2x5x6), rank 12
    for i in range(2):
        for k in range(p):
            t = i * p + k
            for j in range(m):
                U[i * m + j][t] = 1.0
            for j in range(m):
                V[j * p + k][t] = 1.0
            W[k * n + i][t] = 1.0
    
    # Part 2: row 2 (1x5x6), rank 6
    for k in range(p):
        t = 12 + k
        for j in range(m):
            U[2 * m + j][t] = 1.0
        for j in range(m):
            V[j * p + k][t] = 1.0
        W[k * n + 2][t] = 1.0
    
    return {"U": U, "V": V, "W": W}
