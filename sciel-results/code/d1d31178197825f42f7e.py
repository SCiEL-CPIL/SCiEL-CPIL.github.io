"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank)
    - V has shape (m*p, rank)  
    - W has shape (p*n, rank)
    
    The tensor identity is: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For 3x5x6 matrix multiplication, the schoolbook decomposition has rank n*m*p = 90.
    No better-than-schoolbook algorithm is known for these specific dimensions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 90
    
    # Initialize factor matrices with zeros
    U = [[0] * rank for _ in range(n * m)]  # 15 x 90
    V = [[0] * rank for _ in range(m * p)]  # 30 x 90
    W = [[0] * rank for _ in range(p * n)]  # 18 x 90
    
    # Schoolbook decomposition: each term corresponds to one (i,j,k) triple
    # U[i*m + j, c] = 1 if c corresponds to (i,j,k)
    # V[j*p + k, c] = 1 if c corresponds to (i,j,k)
    # W[k*n + i, c] = 1 if c corresponds to (i,j,k)
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
