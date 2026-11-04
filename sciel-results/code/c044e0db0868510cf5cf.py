def construct():
    n, m, p = 3, 4, 6
    rank_schoolbook = n * m * p  # 72
    
    # Try to find a lower rank decomposition
    # Start with a slightly lower rank and see if we can find a valid decomposition
    
    # For now, return the schoolbook decomposition as a baseline
    U = np.zeros((n * m, rank_schoolbook), dtype=np.int32)
    V = np.zeros((m * p, rank_schoolbook), dtype=np.int32)
    W = np.zeros((p * n, rank_schoolbook), dtype=np.int32)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
