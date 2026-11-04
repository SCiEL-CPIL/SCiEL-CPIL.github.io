"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # Rank = n * p = 14 (using M_{1,m,p} decomposition replicated n times)
    rank = n * p  # 14
    
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14
    
    U = [[0] * rank for _ in range(N_U)]
    V = [[0] * rank for _ in range(N_V)]
    W = [[0] * rank for _ in range(N_W)]
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            # U: row i*m + j gets 1 for all j
            for j in range(m):
                U[i * m + j][r] = 1
            # V: row j*p + k gets 1 for all j
            for j in range(m):
                V[j * p + k][r] = 1
            # W: row k*n + i gets 1
            W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
