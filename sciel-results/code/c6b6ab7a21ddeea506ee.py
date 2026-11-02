def construct():
    n, m, p = 4, 4, 9
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for r_prime in range(n):
            r = k * n + r_prime
            # U: row i*m + j
            for i in range(n):
                for j in range(m):
                    if i == r_prime:
                        U[i * m + j][r] = 1
            # V: row j*p + k
            for j in range(m):
                V[j * p + k][r] = 1
            # W: row k*n + i
            for i in range(n):
                if i == r_prime:
                    W[k * n + i][r] = 1
                    
    return {"U": U, "V": V, "W": W}
