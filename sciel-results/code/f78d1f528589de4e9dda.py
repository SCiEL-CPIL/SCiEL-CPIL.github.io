def construct():
    n, m, p = 4, 4, 9
    rank = m
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                if j == r:
                    U[i * m + j, r] = 1
                    
        for j in range(m):
            for k in range(p):
                if j == r:
                    V[j * p + k, r] = 1
                    
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
