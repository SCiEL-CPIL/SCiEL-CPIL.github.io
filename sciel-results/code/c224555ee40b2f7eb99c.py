"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact tensor decomposition for 2x4 by 4x4 matrix multiplication.
    # This uses a rank-8 decomposition derived from treating the two rows independently.
    # The bilinear map rank for 1xn x n x n is n, so for 2xn x n x n it is 2n = 8 when n=4.
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 2 * m  # 8
    
    # U has shape (n*m, rank) = (8, 8)
    # V has shape (m*p, rank) = (16, 8)
    # W has shape (p*n, rank) = (8, 8)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # For each of the 2 rows (i=0,1) and each of the 4 columns of B (k=0..3),
    # we use one rank factor. The r-th rank factor corresponds to row i = r // m and k = r % m.
    # 
    # U[i*m + j, r] = 1 if j == r % m, else 0. (Selects the j-th column of A's row i)
    # V[j*p + k, r] = 1 if k == r % m, else 0. (Selects the k-th column of B)
    # W[k*n + i, r] = 1. (Places the product in position (i,k) of C)
    
    for r in range(rank):
        i = r // m  # row of A (0 or 1)
        k = r % m   # column of B / column of C
        
        for j in range(m):
            # U[i*m + j, r]
            if j == k:
                U[i * m + j][r] = 1.0
            
            # V[j*p + k, r]
            V[j * p + k][r] = 1.0
        
        # W[k*n + i, r]
        W[k * n + i][r] = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
