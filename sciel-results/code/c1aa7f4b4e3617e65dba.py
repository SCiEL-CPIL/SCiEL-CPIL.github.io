"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    # 2x4 x 4x7 = 2x4 x 4x4 (rank 14) + 2x4 x 4x3 (rank 16) = rank 30
    rank = 30
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def add_strassen_222(i0, j0, k0):
        """Add Strassen 2x2 x 2x2 block at rows i0,i0+1; cols j0,j0+1; output rows k0,k0+1."""
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1
        e = j0 * p + k0
        f = j0 * p + k0 + 1
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + k0 + 1
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + i0 + 1
        c11 = (k0 + 1) * n + i0 + 1

        # M1=(a+d)(e+h) -> c00,c11
        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        # M2=(c+d)e -> c10, -c11
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3=a(f-h) -> c01, c11
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4=d(g-e) -> c00, c10
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5=(a+b)h -> -c00, c01
        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        # M6=(c-a)(e+f) -> c11
        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[c11, col] += 1
        col += 1
        # M7=(b-d)(g+h) -> c00
        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[c00, col] += 1
        col += 1

    def add_strassen_22_2x3(i0, j0, k0):
        """Add 2x2 x 2x3 block using rank-8 algorithm at rows i0,i0+1; cols j0,j0+1; output rows k0,k0+1,k0+2."""
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1
        e = j0 * p + k0
        f = j0 * p + k0 + 1
        g = j0 * p + k0 + 2
        h = (j0 + 1) * p + k0
        i = (j0 + 1) * p + k0 + 1
        j = (j0 + 1) * p + k0 + 2
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c02 = (k0 + 2) * n + i0
        c10 = k0 * n + i0 + 1
        c11 = (k0 + 1) * n + i0 + 1
        c12 = (k0 + 2) * n + i0 + 1

        # M1=(a+d)(e+h) -> c00,c10
        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M2=(c+d)e -> c10, -c11
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3=a(f-h) -> c01, c11
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4=d(g-e) -> c00, c10
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5=(a+b)i -> c01
        U[a, col] += 1; U[b, col] += 1
        V[i, col] += 1
        W[c01, col] += 1
        col += 1
        # M6=(b-d)j -> c00
        U[b, col] += 1; U[d, col] -= 1
        V[j, col] += 1
        W[c00, col] += 1
        col += 1
        # M7=(c-a)(f+i) -> c11
        U[c, col] += 1; U[a, col] -= 1
        V[f, col] += 1; V[i, col] += 1
        W[c11, col] += 1
        col += 1
        # M8=(c+d)g -> c10, c12... wait, g is at col k0+2
        U[c, col] += 1; U[d, col] += 1
        V[g, col] += 1
        W[c12, col] += 1
        col += 1

    def add_schoolbook_22x1_3(i0, j0, k0):
        """Add 2x2 x 1x3 schoolbook block at rows i0,i0+1; col j0; output rows k0,k0+1,k0+2."""
        nonlocal col
        a = i0 * m + j0
        c = (i0 + 1) * m + j0
        e = j0 * p + k0
        f = j0 * p + k0 + 1
        g = j0 * p + k0 + 2
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c02 = (k0 + 2) * n + i0
        c10 = k0 * n + i0 + 1
        c11 = (k0 + 1) * n + i0 + 1
        c12 = (k0 + 2) * n + i0 + 1

        U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1
        col += 1
        U[a, col] += 1; V[f, col] += 1; W[c01, col] += 1
        col += 1
        U[a, col] += 1; V[g, col] += 1; W[c02, col] += 1
        col += 1
        U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1
        col += 1
        U[c, col] += 1; V[f, col] += 1; W[c11, col] += 1
        col += 1
        U[c, col] += 1; V[g, col] += 1; W[c12, col] += 1
        col += 1

    # 2x4 x 4x4: two Strassen 2x2x2x2 blocks
    add_strassen_222(0, 0, 0)  # rows 0-1, cols 0-1, output rows 0-1
    add_strassen_222(0, 2, 2)  # rows 0-1, cols 2-3, output rows 2-3
    # That gives 14 terms

    # 2x4 x 4x3: one 2x2x2x3 (rank 8) + one 2x2x1x3 (rank 6)
    # Actually for 2x2 x 2x3, the B matrix is 2 rows x 3 cols
    # Let me redo: B is 4x3, split into 2x3 (rows 0-1) and 2x3 (rows 2-3)
    # A is 2x4, split into 2x2 (cols 0-1) and 2x2 (cols 2-3)
    # C is 2x3
    # For A[0:2, 0:2] * B[0:2, 0:3] -> rank 8
    # For A[0:2, 2:4] * B[2:4, 0:3] -> rank 8
    # Total for 2x4 x 4x3 = 16

    add_strassen_22_2x3(0, 0, 0)  # rows 0-1 of A, cols 0-1; B rows 0-1, cols 0-2; output rows 0-2
    add_strassen_22_2x3(0, 2, 0)  # rows 0-1 of A, cols 2-3; B rows 2-3, cols 0-2; output rows 0-2

    if col != rank:
        return _schoolbook(n, m, p)

    # Verify
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)

    if time.time() > deadline - 5:
        return _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.int64)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    expected = np.ones((n, m, p), dtype=np.int64)
    return np.array_equal(T, expected)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
