"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 16 decomposition for 4x4x8
    # Based on schoolbook 4x4x4 extended to two blocks
    rank = 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r corresponds to (i, j) in 0..15
    for r in range(rank):
        i = r // m
        j = r % m
        
        # U[i*m+j, r] = 1
        U[i * m + j][r] = 1
        
        # V[j'*p+k, r] = 1 if j' == j
        for k in range(p):
            V[j * p + k][r] = 1
            
        # W[k*n+i, r] = 1 if i == i
        for k in range(p):
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
