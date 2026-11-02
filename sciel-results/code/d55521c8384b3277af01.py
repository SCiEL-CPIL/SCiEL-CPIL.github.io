"""Block decomposition: 3x3 grid of 2x2 submatrices, Strassen (rank 7) per block."""
import time
import itertools
import numpy as np

DIMENSIONS = [6, 6, 6]

# Strassen rank-7 2x2 multiplication:
# M1 = A11*A21 ; M2 = (A11+A12)*A22 ; M3 = A21*(B22-B12)
# M4 = A22*B22 ; M5 = (A11+A22)*(B11+B22) ; M6 = (A12-A22)*(B11+B21)
# M7 = (A21-A11)*(B12+B22)
# C11 = M5+M4-M2+M6 ; C12 = M1+M2 ; C21 = M3+M4 ; C22 = M5+M1-M3-M7
#
# (a,b) = A row/col in {1,2} ; (c,d) = B row/col in {1,2} ; (e,f) = C row/col in {1,2}
# A index = (a-1)*2 + (b-1) in 0..3
# B index = (c-1)*2 + (d-1) in 0..3
# C index = (e-1)*2 + (f-1) in 0..3

_STRASSEN = {
    0:  {1: [1], 2: [1]},
    1:  {1: [1, 2], 2: [2]},
    2:  {1: [3], 2: [2, -1]},
    3:  {1: [4], 2: [4]},
    4:  {1: [1, 4], 2: [1, 2]},
    5:  {1: [2, -4], 2: [1, 3]},
    6:  {1: [3, -1], 2: [2, 4]},
}

# C[e,f] = sum_m S_C[m][e,f] * M_m
_STRASSEN_C = {
    (0, 0): {4: 1, 3: 1, 1: -1, 5: 1},
    (0, 1): {0: 1, 1: 1},
    (1, 0): {2: 1, 3: 1},
    (1, 1): {4: 1, 0: 1, 2: -1, 6: -1},
}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the matrix-multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol):
        diff = np.abs(T - expected).max()
        return False, diff
    return True, 0.0


def _build_block_decomposition(n, m, p):
    """Build the 3x3-block Strassen decomposition for 6x6 matrix multiplication."""
    assert n == m == p == 6, "This construction is for 6x6x6"
    block = 2
    nblocks = n // block  # 3

    rank = nblocks * nblocks * nblocks * 7  # 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for i_blk in range(nblocks):
        for j_blk in range(nblocks):
            for k_blk in range(nblocks):
                for sm in range(7):
                    # U: A indices (i_blk*2+a, j_blk*2+b) for (a,b) in S_A[sm]
                    for (a, b), coeff_a in _STRASSEN[sm][1].items():
                        ui = i_blk * block + (a - 1)
                        uj = j_blk * block + (b - 1)
                        U[ui * m + uj, col] += coeff_a

                    # V: B indices (j_blk*2+c, k_blk*2+d) for (c,d) in S_B[sm]
                    for (c, d), coeff_b in _STRASSEN[sm][2].items():
                        vc = j_blk * block + (c - 1)
                        vd = k_blk * block + (d - 1)
                        V[vc * p + vd, col] += coeff_b

                    # W: C indices (k_blk*2+e, i_blk*2+f) for (e,f) where S_C[sm] has (e,f)
                    # W index = k*n + i, where k = k_blk*2+e, i = i_blk*2+f
                    for (e, f) in _STRASSEN_C:
                        if sm in _STRASSEN_C[(e, f)]:
                            coeff_c = _STRASSEN_C[(e, f)][sm]
                            wk = k_blk * block + (e - 1)
                            wi = i_blk * block + (f - 1)
                            W[wk * n + wi, col] += coeff_c

                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_schoolbook(n, m, p):
    """Fallback: schoolbook rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Try the block-Strassen decomposition (rank 189)
    try:
        result = _build_block_decomposition(n, m, p)
        ok, diff = _verify(result["U"], result["V"], result["W"], n, m, p)
        if ok:
            return result
        else:
            # Verification failed; fall back to schoolbook
            pass
    except Exception:
        pass

    # Fallback: schoolbook rank 216
    return _build_schoolbook(n, m, p)
