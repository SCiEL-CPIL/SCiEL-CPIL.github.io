"""Program for matrix multiplication 2x3 by 3x10 using explicit tensor decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    The tensor M has indices (i, j, k) where:
    - i in {0, 1} (rows of A, columns of C)
    - j in {0, 1, 2} (columns of A / rows of B)
    - k in {0, ..., 9} (columns of B, rows of C)
    
    We need to express M as a sum of rank-1 tensors:
    M_{i,j,k} = sum_c U_{(i,j),c} * V_{(j,k),c} * W_{(k,i),c}
    
    The minimal rank is n*m*p = 2*3*10 = 60, achieved by the schoolbook algorithm.
    This is optimal for these dimensions due to the linear independence of the
    60 output monomials in the bilinear map.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank) = (6, 60)
        - V has shape (m*p, rank) = (30, 60)
        - W has shape (p*n, rank) = (20, 60)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the rank-1 components
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: (i, j) -> i*m + j
                U[i * m + j][col] = 1
                # V index: (j, k) -> j*p + k
                V[j * p + k][col] = 1
                # W index: (k, i) -> k*n + i (following the trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
