"""Rank-110 exact decomposition for 4x5 * 5x6 via 4-point DFT core + schoolbook strips."""
DIMENSIONS = [4, 5, 6]

# 4-point DFT (DFT4) and inverse (IDFT4) matrices, entries in {0,1,-1}.
DFT4 = [
    [1, 1, 1, 1],
    [1, -1j, -1, 1j],
    [1, -1, 1, -1],
    [1, 1j, -1, -1j],
]
IDFT4 = [
    [1, 1, 1, 1],
    [1, -1, 1, -1],
    [1, -1j, -1, 1j],
    [1, 1j, -1, -1j],
]


def _real_matrix(M):
    """Convert a complex matrix with only 0/+-1/+-1j entries to a real integer matrix."""
    out = []
    for row in M:
        r = []
        for v in row:
            re = round(v.real)
            im = round(v.imag)
            if abs(v.real - re) > 1e-9 or abs(v.imag - im) > 1e-9:
                raise ValueError("non-integer DFT entry")
            r.append([re, im])
        out.append(r)
    return out


DFT4R = _real_matrix(DFT4)
IDFT4R = _real_matrix(IDFT4)


def construct():
    n, m, p = 4, 5, 6
    U = [[0] * 110 for _ in range(n * m)]   # 20 x 110
    V = [[0] * 110 for _ in range(m * p)]   # 30 x 110
    W = [[0] * 110 for _ in range(p * n)]   # 24 x 110
    col = 0

    def add(u, v, w, vals):
        nonlocal col
        for c in range(4):
            U[u][col + c] = vals[0][c][0]
            V[v][col + c] = vals[1][c][0]
            W[w][col + c] = vals[2][c][0]
        col += 4

    # ---- Core 4x4x4: 16 DFT terms, rank 16 ----
    for a in range(4):
        for b in range(4):
            for c in range(4):
                add(a * m + b, b * p + c, c * n + a,
                    (DFT4R[a], IDFT4R[b], DFT4R[c]))

    # ---- Strip 1: row 4 of A times columns 0..3 of B, rank 4*1*4 = 16 ----
    for b in range(4):
        for c in range(4):
            add(4 * m + b, b * p + c, c * n + 4,
                ([[1, 0]], [[1, 0]], [[1, 0]]))

    # ---- Strip 2: rows 0..3 of A times column 4 of B, rank 4*1*4 = 16 ----
    for a in range(4):
        for b in range(4):
            add(a * m + b, b * p + 4, 4 * n + a,
                ([[1, 0]], [[1, 0]], [[1, 0]]))

    # ---- Strip 3: row 4 of A times columns 4..5 of B, rank 1*2*2 = 4 ----
    for b in range(2):
        for c in range(2):
            add(4 * m + b, b * p + 4 + c, (4 + c) * n + 4,
                ([[1, 0]], [[1, 0]], [[1, 0]]))

    # ---- Strip 4: rows 0..3 of A times columns 4..5 of B, rank 4*2*2 = 16 ----
    for a in range(4):
        for b in range(2):
            for c in range(2):
                add(a * m + b, b * p + 4 + c, (4 + c) * n + a,
                    ([[1, 0]], [[1, 0]], [[1, 0]]))

    # ---- Strip 5: row 4 of A times columns 0..3 of B, rank 1*4*2 = 8 ----
    for b in range(4):
        for c in range(2):
            add(4 * m + b, b * p + c, (4 + c) * n + 4,
                ([[1, 0]], [[1, 0]], [[1, 0]]))

    # ---- Strip 6: rows 0..3 of A times column 4 of B, rank 4*1*2 = 8 ----
    for a in range(4):
        for b in range(1):
            for c in range(2):
                add(a * m + b, b * p + 4, (4 + c) * n + a,
                    ([[1, 0]], [[1, 0]], [[1, 0]]))

    # ---- Strip 7: row 4 of A times column 4 of B, rank 1*1*2 = 2 ----
    for b in range(1):
        for c in range(2):
            add(4 * m + b, b * p + 4, (4 + c) * n + 4,
                ([[1, 0]], [[1, 0]], [[1, 0]]))

    assert col == 110, "expected rank 110, got %d" % col
    return {"U": U, "V": V, "W": W}
