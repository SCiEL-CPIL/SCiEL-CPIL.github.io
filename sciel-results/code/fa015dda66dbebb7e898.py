def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # = 21
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.eye(rank, dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = j * p + k
                U[i * m + j, r] = 1
                W[k * n + i, r] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
