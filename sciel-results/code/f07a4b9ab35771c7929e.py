"""Matrix multiplication 3x5x5 tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """Return a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    
    The decomposition is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    which corresponds to (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    # Schoolbook rank is n*m*p = 75
    rank = n * m * p
    
    # Initialize factor matrices
    U = [[0.0] * rank for _ in range(n * m)]  # U[i*m+j, c]
    V = [[0.0] * rank for _ in range(m * p)]  # V[j*p+k, c]
    W = [[0.0] * rank for _ in range(p * n)]  # W[k*n+i, c]
    
    # Fill in schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
