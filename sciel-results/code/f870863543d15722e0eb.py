# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # The tensor decomposes into p//m = 3 independent 2x3x3 multiplications.
    # Rank of 2x3x3 multiplication tensor is 6.
    # Total rank = 3 * 6 = 18.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 27 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    col = 0
    for block in range(p // m):
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    # U index: i*m + j
                    U[i * m + j][col] = 1
                    # V index: j*p + (block*m + k)
                    V[j * p + (block * m + k)][col] = 1
                    # W index: (block*m + k)*n + i
                    W[(block * m + k) * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
