DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    rank = 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m+j, r] = 1 if (r mod 4) == j, else 0
    for i in range(n):
        for j in range(m):
            for r in range(rank):
                if r % 4 == j:
                    U[i * m + j][r] = 1
    
    # V[j*p+k, r] = 1 if (k mod 4) == r // 4, else 0
    for j in range(m):
        for k in range(p):
            for r in range(rank):
                if k % 4 == r // 4:
                    V[j * p + k][r] = 1
    
    # W[k*n+i, r] = 1 if r == 4*(k mod 4) + j for some j in {0,1}
    # i.e., r in {4*(k mod 4), 4*(k mod 4)+1}
    for k in range(p):
        for i in range(n):
            base = 4 * (k % 4)
            W[k * n + i][base] = 1
            W[k * n + i][base + 1] = 1
    
    return {"U": U, "V": V, "W": W}
