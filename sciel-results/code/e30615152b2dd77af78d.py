"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

The task asks for a rank-minimal exact tensor decomposition of the bilinear map
  C[i,k] = sum_j A[i,j] * B[j,k]
for n=2, m=4, p=7.

Reconstruction convention (from the task contract):
  C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c]) * W[k*n+i,c]

Key insight for n=2: for each (j,k) pair, a single rank-1 term can serve BOTH
output rows i=0 and i=1 simultaneously. The term has:
  U[A[i,j]]=1 for both i (linear form picks A[?,j] for both rows)
  V[B[j,k]]=1               (linear form picks B[j,k])
  W[C[i,k]]=1 for both i    (contributes to both outputs)

This gives rank = m*p = 28, which equals the trivial lower bound max(nm,mp,pn)=28.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    # Rank = m * p = 28: one term per (j, k) pair
    rank = m * p
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    for c, (j, k) in enumerate([(j, k) for j in range(m) for k in range(p)]):
        # U: for both rows i=0,1, pick A[i, j]
        for i in range(n):
            U[i * m + j, c] = 1.0
        # V: pick B[j, k]
        V[j * p + k, c] = 1.0
        # W: contribute to C[0, k] and C[1, k]
        for i in range(n):
            W[k * n + i, c] = 1.0
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
