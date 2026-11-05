"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    The decomposition is of the form:
    C[i, k] = sum_r U[i, r] * V[r, k] ... wait, the interface is U, V, W.
    
    The tensor T has components T[i, j, k] = 1 for the multiplication A[i,j]B[j,k].
    We want T = sum_r U[:, r] \otimes V[:, r] \otimes W[:, r].
    
    Shapes:
    U: (n*m, rank) -> indices (i, j) mapped to i*m + j? 
    The prompt says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    
    Standard mapping:
    U row index: i*m + j (0 <= i < n, 0 <= j < m)
    V row index: j*p + k (0 <= j < m, 0 <= k < p)
    W row index: k*n + i (0 <= k < p, 0 <= i < n) -- Note the trace convention W[k*n+i, r]
    
    For each (i, j, k), we have a term A[i,j] * B[j,k] contributing to C[i,k].
    This corresponds to a rank-1 tensor:
    u has 1 at (i, j)
    v has 1 at (j, k)
    w has 1 at (k, i)
    
    There are n*m*p such terms.
    For n=2, m=3, p=4, rank = 24.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the specific entries to 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
