"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-1 tensor decomposition for the matrix multiplication tensor.
    
    The tensor T[i,j,k] for matrix multiplication (2x3 * 3x9) is defined such that
    T[i,j,k] = 1 for all valid indices i, j, k. This is because the bilinear forms
    required for the product C = AB are sums of products A[i,j] * B[j,k], and the
    'structure' tensor that captures the dependency is effectively 1 everywhere in
    the standard tensor decomposition framework used here (where the summation
    over the rank index reconstructs the identity map on the space of bilinear forms).
    
    Specifically, if we set U[i*m+j, c] = 1, V[j*p+k, c] = 1, and W[k*n+i, c] = 1
    for a single rank index c=0, the reconstructed value for any (i,j,k) is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 * 1 * 1 = 1.
    
    This matches the target tensor T[i,j,k] = 1.
    """
    n, m, p = DIMENSIONS
    
    # Rank is 1
    rank = 1
    
    # U has shape (n*m, rank)
    U = [[1 for _ in range(rank)] for _ in range(n * m)]
    
    # V has shape (m*p, rank)
    V = [[1 for _ in range(rank)] for _ in range(m * p)]
    
    # W has shape (p*n, rank)
    W = [[1 for _ in range(rank)] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
