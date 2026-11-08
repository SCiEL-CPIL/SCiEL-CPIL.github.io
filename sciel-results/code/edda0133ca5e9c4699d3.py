def construct():
    n, m, p = 2, 4, 6
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for r in range(m):
            col = i * m + r
            
            # U
            u_idx = i * m + r
            U[u_idx][col] = 1
            
            # V
            for k in range(p):
                v_idx = r * p + k
                V[v_idx][col] = 1
            
            # W
            for k in range(p):
                w_idx = k * n + i
                W[w_idx][col] = 1
                
    return {"U": U, "V": V, "W": W}
