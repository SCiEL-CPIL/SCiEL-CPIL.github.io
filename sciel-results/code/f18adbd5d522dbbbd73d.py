def construct():
    n, m, p = 2, 3, 10
    # Rank 21 decomposition:
    # Column 0: A[:,0] * B[0,:] (outer product, rank 1)
    # Columns 1..10: A[:,1] * B[1,:] (outer product, rank 10)
    # Columns 11..20: A[:,2] * B[2,:] (outer product, rank 10)
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :] -> rank 1
    for i in range(n):
        U[i * m + 0][col] = 1
    for k in range(p):
        V[0 * p + k][col] = 1
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # Part 2: A[:, 1] * B[1, :] -> rank 10 (one column per k)
    for k in range(p):
        for i in range(n):
            U[i * m + 1][col] = 1
        V[1 * p + k][col] = 1
        for i in range(n):
            W[k * n + i][col] = 1
        col += 1
    
    # Part 3: A[:, 2] * B[2, :] -> rank 10 (one column per k)
    for k in range(p):
        for i in range(n):
            U[i * m + 2][col] = 1
        V[2 * p + k][col] = 1
        for i in range(n):
            W[k * n + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
