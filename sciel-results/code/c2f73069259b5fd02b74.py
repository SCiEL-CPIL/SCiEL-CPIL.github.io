"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    - Use schoolbook (naive) multiplication for the entire 4x4 * 4x9 matrix.
    - Rank is n*m*p = 4*4*9 = 144.
    - This guarantees correctness and avoids the complexity/errors of Strassen's algorithm
      implementation for rectangular blocks.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    rank = n * m * p  # 144
    U = [[0] * rank for _ in range(n * m)]  # 16 x 144
    V = [[0] * rank for _ in range(m * p)]  # 36 x 144
    W = [[0] * rank for _ in range(p * n)]  # 36 x 144
    
    # Schoolbook decomposition
    # For each i, j, k, we have a term U[i,j] * V[j,k] * W[k,i]
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    # Rank index: i * (m * p) + j * p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
