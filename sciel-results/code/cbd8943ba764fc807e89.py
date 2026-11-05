    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Strategy 1: Rank 24 via two 3x3x2 rank 12 decompositions
    dec1 = find_3x3x2_decomp(seed=1)
    if dec1 is not None and time.time() < deadline:
        dec2 = find_3x3x2_decomp(seed=2)
        if dec2 is not None:
            U1, V1, W1 = dec1
            U2, V2, W2 = dec2
            
            U = np.hstack([U1, U2])
            
            V = np.zeros((m * p, 24))
            for j in range(m):
                for k in range(p):
                    if k < 2:
                        V[j * p + k, 0:12] = V1[j * 2 + k, :]
                    else:
                        V[j * p + k, 12:24] = V2[j * 2 + (k - 2), :]
                        
            W = np.zeros((p * n, 24))
            for k in range(p):
                for i in range(n):
                    if k < 2:
                        W[k * n + i, 0:12] = W1[k * n + i, :]
                    else:
                        W[k * n + i, 12:24] = W2[(k - 2) * n + i, :]
                        
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Strategy 2: Fallback to schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
