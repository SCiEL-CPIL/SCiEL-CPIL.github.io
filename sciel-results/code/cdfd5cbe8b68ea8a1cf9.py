"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-1 tensor decomposition for the 2x2 by 2x13 matrix multiplication.
    
    The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T[i,j,k] is 1 for all i, j, k.
    We decompose T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r].
    For rank 1, we can set U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1.
    Then U*V*W = 1 for all indices.
    """
    n, m, p = DIMENSIONS
    rank = 1
    
    # U shape: (n*m, rank)
    U = [[1] for _ in range(n * m)]
    
    # V shape: (m*p, rank)
    V = [[1] for _ in range(m * p)]
    
    # W shape: (p*n, rank)
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
