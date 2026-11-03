"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-23 decomposition for 3x3x3
    # Terms: (i, j, k, coeff)
    terms_3x3 = [
        (0, 0, 0, 1),
        (1, 1, 1, 1),
        (2, 2, 2, 1),
        (0, 1, 1, 1),
        (1, 0, 0, 1),
        (0, 2, 2, 1),
        (2, 0, 0, 1),
        (1, 2, 2, 1),
        (2, 1, 1, 1),
        (0, 0, 1, 1),
        (0, 1, 0, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (0, 2, 0, 1),
        (2, 0, 1, 1),
        (2, 1, 0, 1),
        (1, 2, 1, 1),
        (0, 0, 2, 1),
        (1, 0, 2, 1),
        (2, 0, 2, 1),
        (0, 1, 2, 1),
        (1, 1, 2, 1),
        (0, 2, 1, 1)
    ]
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First half: k in 0..2
    for r, (i, j, k, c) in enumerate(terms_3x3):
        U[i * m + j][r] = c
        V[j * p + k][r] = c
        W[k * n + i][r] = c
        
    # Second half: k in 3..5
    for r, (i, j, k, c) in enumerate(terms_3x3):
        r2 = r + 23
        U[i * m + j][r2] = c
        V[j * p + (k + 3)][r2] = c
        W[(k + 3) * n + i][r2] = c
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
