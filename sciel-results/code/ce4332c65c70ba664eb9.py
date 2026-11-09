"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for matrix multiplication.
    
    For multiplying a (n x m) matrix by a (m x p) matrix, the schoolbook
    decomposition has rank n*m*p. We use the standard trace convention
    where W has index [k*n + i, r] for the output.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank)
        - V has shape (m*p, rank)  
        - W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # Each term c corresponds to a unique (i, j, k) triple
    # U[i*m + j, c] = 1, V[j*p + k, c] = 1, W[k*n + i, c] = 1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
