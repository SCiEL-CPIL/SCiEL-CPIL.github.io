def _build_rank12(n, m, p):
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U is identity
    for idx in range(n * m):
        U[idx][idx] = 1
        
    # V is all 1s
    for idx in range(m * p):
        for c in range(rank):
            V[idx][c] = 1
            
    # W
    # W[k*n + i, c] = 1 if c // m == i and c % m == j?
    # c = i * m + j
    # Row index r = k * n + i
    # For a given c, i = c // m, j = c % m
    # We need W[k*n + i, c] = 1 for all k
    for c in range(rank):
        i = c // m
        j = c % m
        for k in range(p):
            r = k * n + i
            W[r][c] = 1
            
    return {"U": U, "V": V, "W": W}
