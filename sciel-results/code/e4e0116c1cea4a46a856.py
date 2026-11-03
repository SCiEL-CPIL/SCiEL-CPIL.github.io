"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x7 matrix.
    
    The schoolbook algorithm uses rank n*m*p.
    For n=2, m=3, p=7, rank = 42.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over all combinations of indices i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U corresponds to A[i, j]
                # Index in U: i*m + j
                U[i * m + j][col] = 1
                
                # V corresponds to B[j, k]
                # Index in V: j*p + k
                V[j * p + k][col] = 1
                
                # W corresponds to C[i, k]
                # Index in W: k*n + i (following the convention W[k*n+i, r])
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
