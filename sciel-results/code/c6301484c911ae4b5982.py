def construct():
    n, m, p = DIMENSIONS
    
    rank = m
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.ones((p * n, rank), dtype=int)
    
    # U[i * m + j, r] = 1 if j == r
    for r in range(rank):
        for i in range(n):
            U[i * m + r, r] = 1
            
    # V[j * p + l, r] = 1 if j == r
    for r in range(rank):
        for l in range(p):
            V[r * p + l, r] = 1
            
    # W is all 1s
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
