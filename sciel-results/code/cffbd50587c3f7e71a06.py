"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 1
    # U shape (n*m, rank)
    U = [[1] * rank for _ in range(n * m)]
    # V shape (m*p, rank)
    V = [[1] * rank for _ in range(m * p)]
    # W shape (p*n, rank)
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
