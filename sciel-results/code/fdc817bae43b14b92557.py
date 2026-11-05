def construct():
    n, m, p = 3, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate through all combinations of indices
    # This is a new loop structure compared to the nested i,j,k
    idx = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
