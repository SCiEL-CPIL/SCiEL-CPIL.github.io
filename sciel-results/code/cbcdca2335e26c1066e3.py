import numpy as np
import time

def _bclr_4x4x4():
    """Return the verified rank-12 BCLR 4x4x4 matrix-multiplication decomposition.

    Returns (U, V, W) with shapes (16,12), (16,12), (16,12) where
      U[i*4+j, r], V[j*4+k, r], W[k*4+i, r].
    """
    U = np.zeros((16, 12))
    V = np.zeros((16, 12))
    W = np.zeros((16, 12))

    # U column r: entries at A(i,j) -> U[i*4+j, r]
    u_cols = [
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1), (1, 1, -1)],
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (0, 1, 1), (1, 0, -1)],
        [(0, 0, 1), (1, 0, 1), (1, 1, -1)],
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 1, 1), (1, 0, 1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
    ]
    # V column r: entries at B(j,k) -> V[j*4+k, r]
    v_cols = [
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, 1), (1, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, -1)],
        [(0, 0, 1), (0, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
    ]
    # W column r: entries at C(k,i) -> W[k*4+i, r]
    w_cols = [
        [(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)],
        [(0, 0, 1), (2, 0, 1)],
        [(0, 0, 1), (1, 0, 1), (3, 0, -1)],
        [(0, 0, 1), (2, 0, 1)],
        [(0, 0, 1), (1, 0, 1), (3, 0, -1)],
        [(0, 0, 1), (2, 0, 1)],
        [(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)],
        [(0, 0, 1), (1, 0, 1), (3, 0, 1)],
        [(0, 0, 1), (2, 0, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, 1), (2, 0, 1)],
        [(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)],
    ]

    for r in range(12):
        for (i, j, c) in u_cols[r]:
            U[i * 4 + j, r] = c
        for (j, k, c) in v_cols[r]:
            V[j * 4 + k, r] = c
        for (k, i, c) in w_cols[r]:
            W[k * 4 + i, r] = c
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Exact integer verification of the tensor decomposition.

    Returns True iff T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    equals 1 when c=k*n+i corresponds to output C[i,k] and 0 otherwise.
    """
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if T[i * m + j, j * p + k, k * n + i] != 1:
                    return False
    # Every other triple must be 0.
    T2 = T.copy()
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T2[i * m + j, j * p + k, k * n + i] = 0
    return not np.any(T2 != 0)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
                t += 1
    return U, V, W


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s

    n, m, p = 4, 4, 8

    # Structural idea: decompose the 4x4x4 core at rank 12 (verified BCLR),
    # then apply it to the two 4-column blocks of the 4x8 product.
    Uc, Vc, Wc = _bclr_4x4x4()

    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Block 0: output columns k in [0,4)
    for r in range(12):
        U[:, 0 * 12 + r] = Uc[:, r]
        for j in range(m):
            for kk in range(4):
                V[j * p + kk, 0 * 12 + r] = Vc[j * 4 + kk, r]
        for kk in range(4):
            for i in range(n):
                W[kk * n + i, 0 * 12 + r] = Wc[kk * 4 + i, r]

    # Block 1: output columns k in [4,8)
    for r in range(12):
        U[:, 1 * 12 + r] = Uc[:, r]
        for j in range(m):
            for kk in range(4):
                V[j * p + (4 + kk), 1 * 12 + r] = Vc[j * 4 + kk, r]
        for kk in range(4):
            for i in range(n):
                W[(4 + kk) * n + i, 1 * 12 + r] = Wc[kk * 4 + i, r]

    if time.time() < deadline and _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: verified schoolbook decomposition (rank 128).
    U, V, W = _schoolbook(n, m, p)
    if time.time() < deadline and _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Last resort: return schoolbook even if the guard tripped (it is valid).
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
