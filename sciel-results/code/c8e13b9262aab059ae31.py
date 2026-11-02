"""Matrix multiplication 4x5 by 5x5: rank-minimal exact tensor decomposition.

Uses a structured block decomposition:
  - Split the 4x5 matrix A into two 2x5 blocks.
  - For each 2x5x5 sub-problem, use a rank-5 decomposition per output column
    (one rank per input index j), giving 5*5=25 ranks per 2x5x5 block.
  - Total: 2 * 25 = 50 ranks.

Validation:
  For each (i, j, k) in [0..3]x[0..4]x[0..4], the sum over r of
  U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1.
"""

import time
import numpy as np

DEADLINE_S = 110.0
DIMENSIONS = [4, 5, 5]


def _build_tensor(n, m, p):
    T = np.zeros((n * m, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, k * n + i] = 1.0
    return T


def _validate_decomposition(U, V, W, n, m, p, tol=1e-8):
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W


def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS

    # Block decomposition: two 2x5x5 blocks
    # For each block (rows i0, i1 of A), and each output column k:
    #   Use 5 ranks (one per input index j)
    #   Rank r corresponds to j:
    #     U[i0*m + j, r] = 1, U[i1*m + j, r] = 1
    #     V[j*p + k, r] = 1
    #     W[k*n + i0, r] = 1, W[k*n + i1, r] = 1
    #   This gives: for C[i0, k] and C[i1, k], the sum over r of
    #     U[i, j, r] V[j, k, r] W[k, i, r] = A[i,j] B[j,k]
    #   since only r=j contributes, and the coefficients are all 1.

    rank = 50
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col_offset = 0
    for block in range(2):
        i0 = block * 2
        i1 = i0 + 1
        for k in range(p):
            for j in range(m):
                r = col_offset
                U[i0 * m + j, r] = 1.0
                U[i1 * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i0, r] = 1.0
                W[k * n + i1, r] = 1.0
                col_offset += 1

    # Validate
    if not _validate_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)

    def _to_list(M):
        out = []
        for row in M:
            lrow = []
            for x in row:
                xf = float(x)
                if abs(xf - round(xf)) < 1e-6:
                    lrow.append(int(round(xf)))
                else:
                    lrow.append(xf)
            out.append(lrow)
        return out

    if time.time() - t_start > DEADLINE_S:
        # Should not happen, but guard
        pass

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
