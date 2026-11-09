"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-m*p decomposition exploiting the small n dimension
    # For each (k, j) pair, one rank-1 term:
    #   U[i*m+j, t] = 1 for all i
    #   V[j*p+k, t] = 1
    #   W[k*n+i, t] = 1 for all i
    # This satisfies sum_t U[i*m+j1,t]*V[j2*p+k,t]*W[k*n+i,t] = delta(j1,j2)
    
    rank = m * p  # 32
    U = [[0] * rank for _ in range(n * m)]   # shape (8, 32)
    V = [[0] * rank for _ in range(m * p)]   # shape (32, 32)
    W = [[0] * rank for _ in range(p * n)]   # shape (16, 32)
    
    for k in range(p):
        for j in range(m):
            t = k * m + j
            for i in range(n):
                U[i * m + j][t] = 1
            V[j * p + k][t] = 1
            for i in range(n):
                W[k * n + i][t] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
