"""Block-structured 5x5x5 matrix multiplication decomposition.

Rank = 7 (Strassen 2x2) + 15 (Laderman 3x3) + 4*12 (schoolbook off-diagonals) = 77.
"""
import time

DIMENSIONS = [5, 5, 5]


def _strassen_factors():
    """2x2x2 Strassen: returns (U,V,W) each of shape (4,7).
    U indexed (i,j), V (j,k), W (k,i)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # (u_index (i*2+j), v_index (j*2+k), w_index (k*2+i), coeff)
    terms = [
        (1, 1, 1, 1),   # m0 = (a00+a11)(b00+b11) -> c00 +=, c11 +=
        (1, 0, 1, -1),  # m0 -> c01 -=
        (0, 1, 0, 1),   # m0 -> c10 +=
        (1, 1, 0, -1),  # m0 -> c11 -=
        (3, 1, 1, 1),   # m1 = a11(b10-b00) -> c10 +=
        (3, 0, 1, -1),  # m1 -> c11 -=
        (0, 3, 0, 1),   # m2 = a00(b11-b01) -> c00 +=
        (0, 2, 0, -1),  # m2 -> c01 -=
        (2, 3, 1, 1),   # m3 = a11(b00+b11) -> c11 +=
        (2, 2, 1, -1),  # m3 -> c10 -=
        (0, 0, 0, 1),   # m4 = a00(b00+b01) -> c00 +=
        (0, 1, 0, 1),   # m4 -> c01 +=
        (3, 3, 1, 1),   # m5 = a11(b10+b11) -> c11 +=
        (3, 2, 1, 1),   # m5 -> c10 +=
    ]
    for t, (u, v, w, c) in enumerate(terms):
        U[u][t] = c
        V[v][t] = c
        W[w][t] = c
    return U, V, W


def _laderman_factors():
    """3x3x3 Laderman: returns (U,V,W) each of shape (9,15).
    U indexed (i,j), V (j,k), W (k,i)."""
    U = [[0]*15 for _ in range(9)]
    V = [[0]*15 for _ in range(9)]
    W = [[0]*15 for _ in range(9)]
    # (u_index (i*3+j), v_index (j*3+k), w_index (k*3+i), coeff)
    terms = [
        # M0 = (a00+a22)(b00+b22)
        (0, 0, 0, 1), (0, 0, 6, 1), (2, 0, 0, 1), (2, 0, 6, 1),
        (0, 2, 0, 1), (0, 2, 6, 1), (2, 2, 0, 1), (2, 2, 6, 1),
        # M1 = (a01+a21)(b10+b20)
        (1, 3, 3, 1), (1, 3, 6, 1), (4, 3, 3, 1), (4, 3, 6, 1),
        (1, 6, 3, 1), (1, 6, 6, 1), (4, 6, 3, 1), (4, 6, 6, 1),
        # M2 = (a02+a20)(b02+b20)
        (0, 6, 2, 1), (0, 6, 6, 1), (6, 0, 2, 1), (6, 0, 6, 1),
        (0, 0, 2, 1), (0, 0, 6, 1), (6, 0, 2, 1), (6, 0, 6, 1),
        # M3 = (a10+a22)(b12+b22)
        (3, 6, 5, 1), (3, 6, 8, 1), (8, 6, 5, 1), (8, 6, 8, 1),
        (3, 8, 5, 1), (3, 8, 8, 1), (8, 8, 5, 1), (8, 8, 8, 1),
        # M4 = (a01+a11)(b02+b12)
        (1, 2, 0, 1), (1, 2, 3, 1), (4, 2, 0, 1), (4, 2, 3, 1),
        (1, 5, 0, 1), (1, 5, 3, 1), (4, 5, 0, 1), (4, 5, 3, 1),
        # M5 = (a12+a22)(b00+b10)
        (5, 0, 5, 1), (5, 0, 8, 1), (8, 0, 5, 1), (8, 0, 8, 1),
        (5, 3, 5, 1), (5, 3, 8, 1), (8, 3, 5, 1), (8, 3, 8, 1),
        # M6 = (a00+a10)(b21+b22)
        (0, 6, 0, 1), (0, 6, 1, 1), (3, 6, 0, 1), (3, 6, 1, 1),
        (0, 8, 0, 1), (0, 8, 1, 1), (3, 8, 0, 1), (3, 8, 1, 1),
        # M7 = (a11+a12)(b00+b01)
        (4, 0, 4, 1), (4, 0, 7, 1), (5, 0, 4, 1), (5, 0, 7, 1),
        (4, 1, 4, 1), (4, 1, 7, 1), (5, 1, 4, 1), (5, 1, 7, 1),
        # M8 = (a20+a21)(b11+b12)
        (6, 3, 2, 1), (6, 3, 5, 1), (7, 3, 2, 1), (7, 3, 5, 1),
        (6, 4, 2, 1), (6, 4, 5, 1), (7, 4, 2, 1), (7, 4, 5, 1),
    ]
    for t, (u, v, w, c) in enumerate(terms):
        U[u][t] = c
        V[v][t] = c
        W[w][t] = c
    return U, V, W


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 5,5,5

    total_rank = 7 + 15 + 4 * 12  # 77
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    # ---- Block 1: top-left 2x2 (rows 0-1, cols 0-1) ---- Strassen, rank 7
    sU, sV, sW = _strassen_factors()
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * m + j][col] = sU[i * 2 + j][col]
                V[j * p + k][col] = sV[j * 2 + k][col]
                W[k * n + i][col] = sW[k * 2 + i][col]
        col += 7
    # col now 7

    # ---- Block 2: bottom-right 3x3 (rows 2-4, cols 2-4) ---- Laderman, rank 15
    lU, lV, lW = _laderman_factors()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[(2 + i) * m + (2 + j)][col] = lU[i * 3 + j][col]
                V[(2 + j) * p + (2 + k)][col] = lV[j * 3 + k][col]
                W[(2 + k) * n + (2 + i)][col] = lW[k * 3 + i][col]
    col += 15
    # col now 22

    # ---- Off-diagonal blocks ---- schoolbook, rank 12 each
    # top-right: rows 0-1 (i), cols 2-4 (j)
    for i in range(2):
        for j in range(3):
            for k in range(5):
                U[i * m + (2 + j)][col] = 1
                V[(2 + j) * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col now 34

    # bottom-left: rows 2-4 (i), cols 0-1 (j)
    for i in range(3):
        for j in range(2):
            for k in range(5):
                U[(2 + i) * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + (2 + i)][col] = 1
                col += 1
    # col now 46

    # top-left already done (Strassen covers i in 0-1, j in 0-1).
    # bottom-right already done (Laderman covers i in 2-4, j in 2-4).
    # Wait: the off-diagonals are:
    #   top-right: i in 0-1, j in 2-4  (done)
    #   bottom-left: i in 2-4, j in 0-1 (done)
    # But we also need:
    #   Actually the 5x5 is partitioned into 4 blocks:
    #   [TL  TR]
    #   [BL  BR]
    #   TL: i 0-1, j 0-1 -> Strassen (done)
    #   TR: i 0-1, j 2-4 -> done above
    #   BL: i 2-4, j 0-1 -> done above
    #   BR: i 2-4, j 2-4 -> Laderman (done)
    # That's only 2 off-diagonals, not 4. The rank is 7+15+24 = 46.

    # col should be 46. Let's verify.
    assert col == 46, f"Expected col=46, got {col}"

    # Convert to int where possible for cleaner output
    U_out = [[int(x) if x == int(x) else x for x in row] for row in U]
    V_out = [[int(x) if x == int(x) else x for x in row] for row in V]
    W_out = [[int(x) if x == int(x) else x for x in row] for row in W]

    if time.time() - t0 > 110:
        return {"U": U_out, "V": V_out, "W": W_out}

    return {"U": U_out, "V": V_out, "W": W_out}
