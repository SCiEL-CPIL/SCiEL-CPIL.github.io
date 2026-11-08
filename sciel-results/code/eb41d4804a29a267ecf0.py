def construct():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = (i * m + j) * p + k
                # U
                idx_u = i * m + j
                U[idx_u][r] = 1
                # V
                idx_v = j * p + k
                V[idx_v][r] = 1
                # W
                idx_w = k * n + i
                W[idx_w][r] = 1
                
    return {"U": U, "V": V, "W": W}
