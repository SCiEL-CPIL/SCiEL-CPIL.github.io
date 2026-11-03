DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map (i, j, k) to rank index
    # i in [0, n), j in [0, m), k in [0, p)
    # a = i * m + j
    # b = j * p + k
    # c = k * n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
