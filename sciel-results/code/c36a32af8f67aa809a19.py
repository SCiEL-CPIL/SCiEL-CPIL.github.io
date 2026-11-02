"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 25
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(rank):
        j = t // p
        k = t % p
        
        # V is identity-like: V[j*p + k, t] = 1
        V[j * p + k][t] = 1
        
        # U[i*m + j, t] = 1 for i in 0..n-1
        for i in range(n):
            U[i * m + j][t] = 1
            
        # W[k*n + i, t] = 1 for i in 0..n-1
        for i in range(n):
            W[k * n + i][t] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
