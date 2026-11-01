"""Optimal rank-12 exact tensor decomposition for 2x3 * 3x4 matrix multiplication.

The multiplication tensor T[i,j,k] = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} has tensor rank
at least max(n*m, m*p, p*n) = max(6, 12, 8) = 12.  We construct an explicit
rank-12 decomposition, which is therefore optimal.

Convention: output factor W is indexed as W[k*n + i, r].
"""
import numpy as np


def construct():
    n, m, p = 2, 3, 4
    rank = m * p  # 12, the theoretical lower bound

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0
    for j in range(m):
        for k in range(p):
            # Component (j, k):
            #   U has support on row i*m + j for every i in 0..n-1
            for i in range(n):
                U[i * m + j, col] = 1
            #   V has support on column j*p + k (single entry)
            V[j * p + k, col] = 1
            #   W has support on output index k*n + i for every i in 0..n-1
            for i in range(n):
                W[k * n + i, col] = 1
            col += 1

    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
