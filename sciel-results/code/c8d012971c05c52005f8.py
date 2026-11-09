"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 28
    
    # U is (n*m) x rank
    # V is (m*p) x rank
    # W is (p*n) x rank
    
    U = [[1] * rank for _ in range(n * m)]
    
    V = [[0] * rank for _ in range(m * p)]
    for r in range(rank):
        V[r][r] = 1
        
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
