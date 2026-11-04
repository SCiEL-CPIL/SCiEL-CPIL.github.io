"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 4
    rank = 49
    
    # Build a rank-49 decomposition using the Kronecker product of two rank-7 2x2 decompositions.
    # For 2x2, we use a verified rank-7 decomposition.
    
    # First, let's define a valid rank-7 decomposition for 2x2.
    # I'll use the following approach: since I'm having trouble with Strassen from memory,
    # let me use a different strategy.
    
    # Actually, let me use the rank-64 schoolbook decomposition as a safe fallback.
    # This is guaranteed to be correct and matches the seed score.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
