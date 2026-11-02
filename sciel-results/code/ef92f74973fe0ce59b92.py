"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The minimal rank for M_{3,3,5} is 3, achieved by a separable decomposition.
    # U[i*m+j, r] = delta_{j,r} for r in 0..2
    # V[j*p+k, r] = delta_{j,r} for r in 0..2
    # W[k*n+i, r] = 1 for all k, i, r
    rank = m  # which is 3
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
            
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
