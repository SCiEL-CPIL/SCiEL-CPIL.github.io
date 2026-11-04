DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = min(n, m, p)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            U[idx][i] = 1
            
    # Fill V
    for j in range(m):
        for k in range(p):
            idx = j * p + k
            for r in range(rank):
                V[idx][r] = 1
                
    # Fill W
    for k in range(p):
        for i in range(n):
            idx = k * n + i
            for r in range(rank):
                W[idx][r] = 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }
