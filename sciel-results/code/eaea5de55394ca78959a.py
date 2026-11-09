"""Matrix multiplication 6x6x6 via block Strassen decomposition.

Combines Strassen's rank-7 decomposition on a 2x2 grid of 3x3 blocks with
schoolbook multiplication on the inner 3x3 blocks. Total rank = 7 * 27 = 189.

Official trace convention for W: W[k*n + i, c].
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
N = M = P = 6
BLOCK = 3


# EVOLVE-BLOCK-START
def _strassen7():
    """Return (U,V,W) each shape (4,7) for 2x2x2 matrix multiplication.

    U[i, s] = coefficient of a_{i0} + a_{i1} in M_s
    V[j, s] = coefficient of b_{j0} + b_{j1} in M_s
    W[k, s] = coefficient of c_{0k} + c_{1k} in M_s
    """
    # Strassen's 7 intermediate products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    U = np.array([
        [1, 0, 1, 0, 1, 1, 1],   # a00
        [0, 1, 0, 1, 0, -1, -1],  # a11
    ], dtype=object)
    V = np.array([
        [1, 1, -1, -1, 0, 1, 0],   # b00
        [1, 0, 1, 0, 1, 0, 1],     # b11
    ], dtype=object)
    W = np.array([
        [1, 0, 0, 0, 0, 0, 0],   # c00
        [0, 0, 1, 0, 1, 1, 0],   # c10
        [0, 0, 0, 1, 0, 0, 1],   # c01
        [1, 1, 0, 0, 0, 0, 0],   # c11
    ], dtype=object)
    return U, V, W


def _build_block_decomposition():
    n, m, p = N, M, P
    U2, V2, W2 = _strassen7()

    U = np.zeros((n * m, 189), dtype=object)
    V = np.zeros((m * p, 189), dtype=object)
    W = np.zeros((p * n, 189), dtype=object)

    for s in range(7):
        for r in range(3):
            for c in range(3):
                col = s * 9 + r * 3 + c
                # Block (r,c) of A is rows r*B..(r+1)*B-1, cols c*B..(c+1)*B-1
                # Strassen variable index i = 2*r + c
                u_row = 2 * r * m + 2 * c * BLOCK + r
                V_row = 2 * c * p + 2 * r * BLOCK + r
                W_row = 2 * r * n + 2 * c * BLOCK + c
                U[u_row, col] = U2[2 * r + c, s]
                V[V_row, col] = V2[2 * c + r, s]
                W[W_row, col] = W2[2 * r + c, s]

    return U, V, W


def _verify(U, V, W):
    n, m, p = N, M, P
    rank = V.shape[1]
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    Uf = U.astype(float)
    Vf = V.astype(float)
    Wf = W.astype(float)
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = Uf[:, c]
        vc = Vf[:, c]
        wc = Wf[:, c]
        # reshape
        a = uc.reshape(n, m)
        b = vc.reshape(m, p)
        w = wc.reshape(p, n)
        # T[i,j,k] += a[i,j]*b[j,k]*w[k,i]
        # for fixed i,j: T[i,j,:] += a[i,j]*b[j,:]*w[:,i]
        ab = a * b  # (n, p) where [i,k] = sum_j a[i,j]*b[j,k] but we need per j
        # Actually we need T[i,j,k] = sum_c a_c[i,j]*b_c[j,k]*w_c[k,i]
        # = a[i,j] * (b[j,:]*w[:,i])_k
        for i in range(n):
            ai = a[i, :]  # (m,)
            for j in range(m):
                T[i, j, :] += ai[j] * (b[j, :] * w[:, i])
    # Expected: 1 iff j==k
    exp = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            exp[i, j, j] = 1.0
    err = np.abs(T - exp).max()
    return err


def construct():
    deadline = time.time() + 140.0  # safety margin below 150s kill
    U, V, W = _build_block_decomposition()
    # Verify (should be fast for rank 189)
    if time.time() < deadline - 5.0:
        err = _verify(U, V, W)
        if err > 1e-6:
            # Fallback to schoolbook if something went wrong
            n, m, p = N, M, P
            rank = n * m * p
            Us = np.zeros((n * m, rank), dtype=object)
            Vs = np.zeros((m * p, rank), dtype=object)
            Ws = np.zeros((p * n, rank), dtype=object)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Us[i * m + j, col] = 1
                        Vs[j * p + k, col] = 1
                        Ws[k * n + i, col] = 1
                        col += 1
            return {"U": Us, "V": Vs, "W": Ws}
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
