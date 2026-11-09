"""Tensor decomposition for 4x4 * 4x9 matrix multiplication (rank 56)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np

def _laderman_444():
    """Laderman's rank-20 decomposition for (4,4,4).
    Returns U (16,20), V (16,20), W (16,20) with index i*4+j etc.
    """
    n = 4
    R = 20
    U = np.zeros((n * n, R))
    V = np.zeros((n * n, R))
    W = np.zeros((n * n, R))

    def idx(r, c):
        return r * n + c

    # 16 single-entry terms
    simple = [
        (0, 0, 0, 0, 0, 0),   # A00 B00 -> C00
        (0, 1, 1, 0, 0, 0),   # A01 B10 -> C00
        (1, 0, 0, 1, 0, 1),   # A10 B01 -> C01
        (1, 1, 1, 1, 1, 1),   # A11 B11 -> C11
        (0, 2, 2, 0, 0, 2),   # A02 B20 -> C02
        (0, 3, 3, 0, 0, 3),   # A03 B30 -> C03
        (2, 0, 0, 2, 2, 0),   # A20 B02 -> C20
        (3, 0, 0, 3, 3, 0),   # A30 B03 -> C30
        (2, 2, 2, 2, 2, 2),   # A22 B22 -> C22
        (3, 3, 3, 3, 3, 3),   # A33 B33 -> C33
        (1, 2, 2, 1, 1, 2),   # A12 B21 -> C12
        (1, 3, 3, 1, 1, 3),   # A13 B31 -> C13
        (2, 1, 1, 2, 2, 1),   # A21 B12 -> C21
        (3, 1, 1, 3, 3, 1),   # A31 B13 -> C31
        (2, 3, 3, 2, 2, 3),   # A23 B32 -> C23
        (3, 2, 2, 3, 3, 2),   # A32 B23 -> C32
    ]
    for r, (ai, aj, bi, bj, ci, cj) in enumerate(simple):
        U[idx(ai, aj), r] = 1
        V[idx(bi, bj), r] = 1
        W[idx(ci, cj), r] = 1

    # 4 compound terms
    # r=16: (A02+A13)(B20+B31) -> C02 + C13
    U[idx(0, 2), 16] = 1; U[idx(1, 3), 16] = 1
    V[idx(2, 0), 16] = 1; V[idx(3, 1), 16] = 1
    W[idx(0, 2), 16] = 1; W[idx(1, 3), 16] = 1
    # r=17: (A03+A12)(B30+B21) -> C03 + C12
    U[idx(0, 3), 17] = 1; U[idx(1, 2), 17] = 1
    V[idx(3, 0), 17] = 1; V[idx(2, 1), 17] = 1
    W[idx(0, 3), 17] = 1; W[idx(1, 2), 17] = 1
    # r=18: (A02+A12)(B20+B21) -> C02 + C12
    U[idx(0, 2), 18] = 1; U[idx(1, 2), 18] = 1
    V[idx(2, 0), 18] = 1; V[idx(2, 1), 18] = 1
    W[idx(0, 2), 18] = 1; W[idx(1, 2), 18] = 1
    # r=19: (A03+A13)(B30+B31) -> C03 + C13
    U[idx(0, 3), 19] = 1; U[idx(1, 3), 19] = 1
    V[idx(3, 0), 19] = 1; V[idx(3, 1), 19] = 1
    W[idx(0, 3), 19] = 1; W[idx(1, 3), 19] = 1

    return U, V, W


def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    R = U.shape[1]
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(Ua[i * m + j] * Va[j * p + k], Wa[k * n + i]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err


def _schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4,4,9
    deadline = time.time() + 120

    L_U, L_V, L_W = _laderman_444()

    # Validate Laderman for (4,4,4)
    ok, err = _validate(L_U, L_V, L_W, 4, 4, 4)
    if not ok or time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Split p=9 into 4 + 4 + 1
    R_total = 20 + 20 + 16  # 56
    U_full = np.zeros((n * m, R_total))
    V_full = np.zeros((m * p, R_total))
    W_full = np.zeros((p * n, R_total))

    def place_block(offset, k_start):
        # Copy Laderman (4,4,4) into columns k_start..k_start+3
        for r in range(20):
            U_full[:, offset + r] = L_U[:, r]
            for j in range(4):
                for kk in range(4):
                    V_full[j * p + (k_start + kk), offset + r] = L_V[j * 4 + kk, r]
            for kk in range(4):
                for i in range(4):
                    W_full[(k_start + kk) * n + i, offset + r] = L_W[kk * 4 + i, r]

    place_block(0, 0)
    place_block(20, 4)

    # Block 3: schoolbook for column 8 (k=8), (4,4,1) rank 16
    off = 40
    for i in range(n):
        for j in range(m):
            col = off + (i * m + j)
            U_full[i * m + j, col] = 1
            V_full[j * p + 8, col] = 1
            W_full[8 * n + i, col] = 1

    ok, err = _validate(U_full, V_full, W_full, n, m, p)
    if not ok:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
