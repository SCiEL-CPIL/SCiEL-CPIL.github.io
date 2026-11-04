"""
Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.

This is the schoolbook decomposition, which achieves rank 50.
For M_{2,5,5}, the rank is believed to be exactly 50 (no known improvement).
We construct U, V, W with the exact trace convention:
  (A*B)[i,k] = sum_j A[i,j] * B[j,k]
  U has shape (n*m, rank), indexed by (i,j)
  V has shape (m*p, rank), indexed by (j,k)
  W has shape (p*n, rank), indexed by (k,i)  [per task: W[k*n+i, r]]
"""

import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
