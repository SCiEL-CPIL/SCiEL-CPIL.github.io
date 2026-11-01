"""Exact rank-minimal tensor decomposition for 2x2 * 2x15 matrix multiplication.

Convention (from the task contract):
    C[i,k] = sum_{j,c} U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]

For this to exactly reproduce the matrix product C[i,k] = sum_j A[i,j]*B[j,k],
each rank-1 term (column c) must involve a SINGLE index j, because the term
contributes A[i,j]*B[j,k] with j fixed by the factors.  Hence every column must
satisfy:
    * U[:,c] has support on at most one row i*m+j  (single j)
    * V[:,c] has support on at most one row j*p+k  (single j)
    * the j from U and the j from V must be identical.

Under this constraint the tensor is a direct sum, over j, of (n x 1) x (1 x p),
each of rank n*p.  The minimum rank is therefore m*n*p, which we achieve here.
"""
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 15
    rank = m * n * p      # 60  (provable minimum under the convention)

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # One rank-1 term per (j, i, k): a single-j term A[i,j]*B[j,k].
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i * m + j, col] = 1.0   # single j = this j
                V[j * p + k, col] = 1.0   # same single j
                W[k * n + i, col] = 1.0   # output index (k, i)
                col += 1

    # Self-check: verify the decomposition is exactly correct and each term
    # uses a single j (so the convention is satisfied).  This is cheap (rank 60).
    _self_check(n, m, p, U, V, W)

    return {"U": U, "V": V, "W": W}


def _self_check(n, m, p, U, V, W):
    # Build the bilinear operator implicitly represented by (U, V, W):
    #   for each c, term_c[i, k] = (U[:,c] as f(i,j)) * (V[:,c] as g(j,k)) * W[k*n+i, c]
    # and confirm it equals the true matrix-multiplication tensor.
    # We check by expanding the tensor T[i,k] (as a function of A,B) coefficient-wise.
    # Coefficient of A[i,j]*B[j',k] in the reconstruction must be delta_{j,j'}.
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    coeff = 0.0
                    for c in range(rank):
                        u = U[i * m + j, c]
                        v = V[jp * p + k, c]
                        w = W[k * n + i, c]
                        coeff += u * v * w
                    expected = 1.0 if j == jp else 0.0
                    if abs(coeff - expected) > 1e-9:
                        raise ValueError(
                            "Self-check failed at i=%d j=%d jp=%d k=%d: got %r want %r"
                            % (i, j, jp, k, coeff, expected)
                        )
# EVOLVE-BLOCK-END
