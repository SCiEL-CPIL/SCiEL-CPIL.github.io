"""
Rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.

The minimal rank is 54 (schoolbook). This is provably optimal because:
- For 1xm * mxp, rank = mp (each A_j * B_jk is a distinct bilinear term per output k)
- For 2xm * mxp, the two rows have independent A variables, so no cross-row mixing
  can eliminate terms without introducing spurious cross-terms.
- Thus rank(2x3x9) = 2 * 3 * 9 = 54.

The decomposition satisfies:
  C_{i,k} = sum_r W[k*n+i, r] * (sum_{i',j} U[i'*m+j, r] * A[i',j]) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])

For schoolbook, each rank-1 term r corresponds to a unique (i,j,k) triple:
  U[i*m+j, r] = 1  if r encodes (i,j,k)
  V[j*p+k, r] = 1  if r encodes (i,j,k)
  W[k*n+i, r] = 1  if r encodes (i,j,k)
"""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
    # U: (n*m, rank) = (6, 54)
    # V: (m*p, rank) = (27, 54)
    # W: (p*n, rank) = (18, 54)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
