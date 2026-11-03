"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the tensor decomposition exploiting the Kronecker-like structure.
    Decomposes the n x m matrix into n blocks of 1 x m.
    For each block, performs the 1 x m by m x p multiplication.
    The rank of 1 x m x p multiplication is m * p.
    Total rank is n * m * p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over the 'Kronecker' components:
    # i corresponds to the row of the n x m matrix (the 1 x m block)
    for i in range(n):
        # For each row, we decompose the 1 x m by m x p multiplication.
        # This sub-problem has rank m * p.
        for j in range(m):
            for k in range(p):
                # U index: i*m + j (Element (i, j) of the n x m matrix)
                U[i * m + j][col] = 1
                # V index: j*p + k (Element (j, k) of the m x p matrix)
                V[j * p + k][col] = 1
                # W index: k*n + i (Trace convention: W[k*n+i, r])
                # This maps the result to row i, column k of the n x p matrix
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
