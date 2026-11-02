def construct():
    n, m, p = 2, 3, 4
    rank = m * p  # 12
    U = [[0] * rank for _ in range(n * m)]  # 6 x 12
    V = [[0] * rank for _ in range(m * p)]  # 12 x 12
    W = [[0] * rank for _ in range(p * n)]  # 8 x 12
    
    for c in range(rank):
        j_prime = c // p  # 0,0,0,0,1,1,1,1,2,2,2,2
        k = c % p         # 0,1,2,3,0,1,2,3,0,1,2,3
        
        # U[i*m + j, c] = 1 if j == j_prime
        for i in range(n):
            U[i * m + j_prime][c] = 1
        
        # V[j'*p + k, c] = 1
        V[j_prime * p + k][c] = 1
        
        # W[k*n + i, c] = 1 if k == k (always true for the given k)
        for i in range(n):
            W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
