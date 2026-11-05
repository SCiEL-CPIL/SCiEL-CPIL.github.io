"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m  # 15
    
    # U: shape (n*m, rank) = (15, 15), U[i*m+j, c] = 1 if c == i*m+j else 0
    U = [[0] * rank for _ in range(n * m)]
    for i in range(n):
        for j in range(m):
            U[i * m + j][i * m + j] = 1
    
    # V: shape (m*p, rank) = (30, 15), all ones
    V = [[1] * rank for _ in range(m * p)]
    
    # W: shape (p*n, rank) = (18, 15), all ones
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
