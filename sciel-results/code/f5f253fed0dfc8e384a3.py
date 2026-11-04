"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    # Peeling/slab decomposition: recursively peel off columns of B.
    # For 2x3xp, peel one column: 2x3x1 (rank 6) + 2x3x(p-1)
    # Total rank = 7 * 6 = 42 (schoolbook), but we implement it via peeling for clarity.
    
    rank = n * m * p  # 42
    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 21 x rank
    W = [[0] * rank for _ in range(p * n)]  # 14 x rank
    
    col = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
