DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = n * p * m  # Wait, rank is sum over k of m terms. So p * m.
    # m=4, p=9. rank = 36.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for j in range(m):
            # r = col
            for i in range(n):
                U[i * m + j][col] = 1
                W[k * n + i][col] = 1
            V[j * p + k][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
