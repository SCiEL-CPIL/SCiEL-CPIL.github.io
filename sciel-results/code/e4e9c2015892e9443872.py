"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for 3x3 matrix multiplied by 3x7 matrix.
    Rank is n*m*p = 3*3*7 = 63.
    U shape: (3*3, 63)
    V shape: (3*7, 63)
    W shape: (7*3, 63)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # Using lists of lists for compatibility and simplicity, as per seed
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
