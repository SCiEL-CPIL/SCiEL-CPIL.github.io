"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(rank):
        i = t // (m * p)
        rem = t % (m * p)
        j = rem // p
        k = rem % p
        
        U[i * m + j][t] = 1
        V[j * p + k][t] = 1
        W[k * n + i][t] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
