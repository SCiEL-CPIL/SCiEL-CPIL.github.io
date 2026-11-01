"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * p  # 26
    U = [[0] * rank for _ in range(n * m)]  # 4 x 26
    V = [[0] * rank for _ in range(m * p)]  # 26 x 26
    W = [[0] * rank for _ in range(p * n)]  # 26 x 26
    
    for k in range(p):
        # Term 2k: uses A[i,0] and B[0,k], contributes to C[i,k] for i=0,1
        t0 = 2 * k
        for i in range(n):
            U[i * m + 0][t0] = 1  # A[i, 0]
        V[0 * p + k][t0] = 1  # B[0, k]
        for i in range(n):
            W[k * n + i][t0] = 1  # C[k, i] -> index k*n+i
        
        # Term 2k+1: uses A[i,1] and B[1,k], contributes to C[i,k] for i=0,1
        t1 = 2 * k + 1
        for i in range(n):
            U[i * m + 1][t1] = 1  # A[i, 1]
        V[1 * p + k][t1] = 1  # B[1, k]
        for i in range(n):
            W[k * n + i][t1] = 1  # C[k, i] -> index k*n+i
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
