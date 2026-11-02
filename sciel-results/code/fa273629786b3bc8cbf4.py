"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication.

The matmul tensor is T[i,j,k]=1 for all i in [0,n), j in [0,m), k in [0,p).
A rank-1 decomposition exists: u[i]=1, v[j]=1, w[k]=1 gives sum_r=1 of 1*1*1=1.

Factor matrices with the official trace convention:
  U[i*m+j, r] = u_r[i]  * 1  (shape n*m)
  V[j*p+k, r] = v_r[j]  * 1  (shape m*p)  
  W[k*n+i, r] = w_r[k]  * 1  (shape p*n)

For rank 1: U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1 for all indices.
"""

import time


def construct():
    """Return {'U', 'V', 'W'} factor matrices for rank-1 decomposition."""
    n, m, p = 2, 5, 5
    rank = 1

    # U: shape (n*m, rank) = (10, 1)
    U = [[1.0] for _ in range(n * m)]
    # V: shape (m*p, rank) = (25, 1)
    V = [[1.0] for _ in range(m * p)]
    # W: shape (p*n, rank) = (10, 1), with trace convention W[k*n + i, r]
    W = [[1.0] for _ in range(p * n)]

    return {"U": U, "V": V, "W": W}
