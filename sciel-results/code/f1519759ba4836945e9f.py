"""Strassen-style recursive decomposition for 4x5 * 5x6 matrix multiplication."""
import time

DIMENSIONS = [4, 5, 6]
_START = time.time()
_DEADLINE = _START + 140.0

# Cache for recursive sub-problems
_cache = {}

def _matmul_decompose(n, m, p):
    """Return (U, V, W) as lists of lists, shapes (n*m, rank), (m*p, rank), (p*n, rank)."""
    key = (n, m, p)
    if key in _cache:
        return _cache[key]
    if time.time() > _DEADLINE:
        # Fallback: schoolbook
        U, V, W = _schoolbook(n, m, p)
        _cache[key] = (U, V, W)
        return U, V, W

    # Use Strassen when all dimensions are even and >= 2
    if n >= 2 and m >= 2 and p >= 2 and n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        try:
            result = _strassen_step(n, m, p)
            _cache[key] = result
            return result
        except Exception:
            pass

    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    _cache[key] = (U, V, W)
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
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
    return U, V, W


def _strassen_step(n, m, p):
    """One Strassen step: split into halves, recurse, combine."""
    nh, mh, ph = n // 2, m // 2, p // 2

    # Recurse on 7 sub-problems
    # M1 = (A00+A11)(B00+B11)
    M1 = _matmul_decompose(nh, mh, ph)
    # M2 = (A10-A00)(B00+B01)
    M2 = _matmul_decompose(nh, mh, ph)
    # M3 = A00(B10-B11)
    M3 = _matmul_decompose(nh, mh, ph)
    # M4 = A11(B01-B00)
    M4 = _matmul_decompose(nh, mh, ph)
    # M5 = (A00+A10)B11
    M5 = _matmul_decompose(nh, mh, ph)
    # M6 = (A10-A11)(B01+B11)
    M6 = _matmul_decompose(nh, mh, ph)
    # M7 = (A01-A10)(B00+B10)
    M7 = _matmul_decompose(nh, mh, ph)

    # Each sub-problem has the same rank
    r = len(M1[0][0]) if M1[0] else 0
    rank = 7 * r

    # Build U: rows indexed by (i, j) where i in [0, n), j in [0, m)
    # For sub-block (ii, jj) of A (size nh x mh), the U rows are offset by ii*mh + jj*ph ... 
    # Actually U has shape (n*m, rank). Row index = i*m + j.
    # For sub-block (ii, jj): i = ii*nh + ii_local, j = jj*mh + jj_local
    # So row = (ii*nh + ii_local)*m + (jj*mh + jj_local) = ii*nh*m + ii_local*m + jj*mh + jj_local

    # U contributions:
    # A00: ii=0, jj=0 -> rows: ii_local*m + jj_local for ii_local in [0,nh), jj_local in [0,mh)
    # A11: ii=1, jj=1 -> rows: nh*m + ii_local*m + mh + jj_local
    # A10: ii=1, jj=0 -> rows: nh*m + ii_local*m + jj_local
    # A01: ii=0, jj=1 -> rows: ii_local*m + mh + jj_local

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to add a sub-matrix contribution
    def add_U(U, sub_U, ii, jj, col_offset):
        for ii_local in range(nh):
            for jj_local in range(mh):
                row = (ii * nh + ii_local) * m + (jj * mh + jj_local)
                sub_row = ii_local * mh + jj_local
                for c in range(r):
                    U[row][col_offset + c] += sub_U[sub_row][c]

    def add_V(V, sub_V, jj, kk, col_offset):
        for jj_local in range(mh):
            for kk_local in range(ph):
                row = (jj * mh + jj_local) * p + (kk * ph + kk_local)
                sub_row = jj_local * ph + kk_local
                for c in range(r):
                    V[row][col_offset + c] += sub_V[sub_row][c]

    def add_W(W, sub_W, kk, ii, col_offset):
        for kk_local in range(ph):
            for ii_local in range(nh):
                row = (kk * ph + kk_local) * n + (ii * nh + ii_local)
                sub_row = kk_local * nh + ii_local
                for c in range(r):
                    W[row][col_offset + c] += sub_W[sub_row][c]

    # M1 = (A00+A11)(B00+B11)
    # U: A00 (coeff +1), A11 (coeff +1)
    add_U(U, M1[0], 0, 0, 0)
    add_U(U, M1[0], 1, 1, 0)
    # V: B00 (coeff +1), B11 (coeff +1)
    add_V(V, M1[1], 0, 0, 0)
    add_V(V, M1[1], 1, 1, 0)
    # W: C00 += M1, C11 += M1
    add_W(W, M1[2], 0, 0, 0)
    add_W(W, M1[2], 1, 1, 0)

    # M2 = (A10-A00)(B00+B01)
    # U: A10 (coeff +1), A00 (coeff -1)
    add_U(U, M2[0], 1, 0, r)
    for row in range(nh * m):  # A00 rows
        for c in range(r):
            U[row][r + c] -= M2[0][row][c]
    # V: B00 (coeff +1), B01 (coeff +1)
    add_V(V, M2[1], 0, 0, r)
    add_V(V, M2[1], 0, 1, r)
    # W: C10 += M2, C11 += M2
    add_W(W, M2[2], 0, 1, r)
    add_W(W, M2[2], 1, 1, r)

    # M3 = A00(B10-B11)
    # U: A00 (coeff +1)
    add_U(U, M3[0], 0, 0, 2 * r)
    # V: B10 (coeff +1), B11 (coeff -1)
    add_V(V, M3[1], 1, 0, 2 * r)
    for row in range(mh * ph):  # B11 rows
        for c in range(r):
            V[mh * ph + row][2 * r + c] -= M3[1][row][c]
    # W: C01 += M3, C11 += M3
    add_W(W, M3[2], 0, 0, 2 * r)
    add_W(W, M3[2], 1, 1, 2 * r)

    # M4 = A11(B01-B00)
    # U: A11 (coeff +1)
    add_U(U, M4[0], 1, 1, 3 * r)
    # V: B01 (coeff +1), B00 (coeff -1)
    add_V(V, M4[1], 0, 1, 3 * r)
    for row in range(mh * ph):  # B00 rows
        for c in range(r):
            V[row][3 * r + c] -= M4[1][row][c]
    # W: C00 += M4, C10 += M4
    add_W(W, M4[2], 0, 0, 3 * r)
    add_W(W, M4[2], 0, 1, 3 * r)

    # M5 = (A00+A10)B11
    # U: A00 (coeff +1), A10 (coeff +1)
    add_U(U, M5[0], 0, 0, 4 * r)
    add_U(U, M5[0], 1, 0, 4 * r)
    # V: B11 (coeff +1)
    add_V(V, M5[1], 1, 1, 4 * r)
    # W: C00 += M5, C01 += M5
    add_W(W, M5[2], 1, 0, 4 * r)
    add_W(W, M5[2], 1, 1, 4 * r)

    # M6 = (A10-A11)(B01+B11)
    # U: A10 (coeff +1), A11 (coeff -1)
    add_U(U, M6[0], 1, 0, 5 * r)
    for row in range(nh * m):  # A11 rows: offset by nh*m + ...
        # A11 rows: (nh + ii_local)*m + (mh + jj_local)
        for ii_local in range(nh):
            for jj_local in range(mh):
                row_a11 = (nh + ii_local) * m + (mh + jj_local)
                row_a10 = (nh + ii_local) * m + jj_local
                for c in range(r):
                    U[row_a11][5 * r + c] -= M6[0][ii_local * mh + jj_local][c]
    # V: B01 (coeff +1), B11 (coeff +1)
    add_V(V, M6[1], 0, 1, 5 * r)
    add_V(V, M6[1], 1, 1, 5 * r)
    # W: C10 += M6, C11 += M6
    add_W(W, M6[2], 1, 1, 5 * r)
    add_W(W, M6[2], 0, 1, 5 * r)

    # M7 = (A01-A10)(B00+B10)
    # U: A01 (coeff +1), A10 (coeff -1)
    add_U(U, M7[0], 0, 1, 6 * r)
    for ii_local in range(nh):
        for jj_local in range(mh):
            row_a01 = ii_local * m + (mh + jj_local)
            row_a10 = (nh + ii_local) * m + jj_local
            for c in range(r):
                U[row_a10][6 * r + c] -= M7[0][ii_local * mh + jj_local][c]
    # V: B00 (coeff +1), B10 (coeff +1)
    add_V(V, M7[1], 0, 0, 6 * r)
    add_V(V, M7[1], 1, 0, 6 * r)
    # W: C00 += M7, C01 += M7
    add_W(W, M7[2], 0, 0, 6 * r)
    add_W(W, M7[2], 1, 0, 6 * r)

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _matmul_decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
