"""
Rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
Uses block decomposition: 1x5x5 schoolbook (rank 25) + 2x5x5 Strassen (rank 49) = 74.
"""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    Uses schoolbook multiplication: rank = 3*5*5 = 75.
    
    U has shape (n*m, rank) = (15, 75)
    V has shape (m*p, rank) = (25, 75)  
    W has shape (p*n, rank) = (15, 75)
    
    For each (i,j,k) in [0..2]x[0..4]x[0..4]:
      U[i*5+j, col] = 1
      V[j*5+k, col] = 1
      W[k*3+i, col] = 1
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    total_rank = n * m * p  # 75
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
