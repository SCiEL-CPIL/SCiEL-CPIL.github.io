"""Rank-minimal decomposition for 4x4 @ 4x5 multiplication."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen macros for a 2x2 block of the C matrix:
    # each macro (u, v, w) are length-4 vectors (local 2x2 indices)
    # M1 = A00*(B00+B11), M2 = (A10+A11)*B00, M3 = A00*(B10-B11),
    # M4 = A11*(B01-B10), M5 = (A00+A11)*(B00+B11),
    # M6 = (A10-A00)*B10, M7 = (A01-A11)*B01
    M1 = ([(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
          [(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
          [(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
    M2 = ([(0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0)],
          [(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
          [(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
    M3 = ([(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
          [(0, 0, 0), (0, 0, 0), (1, 0, 0), (-1, 0, 0)],
          [(0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 0, 0)])
    M4 = ([(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0)],
          [(1, 0, 0), (0, 0, 0), (-1, 0, 0), (0, 0, 0)],
          [(0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0)])
    M5 = ([(1, 0, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0)],
          [(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
          [(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0)])
    M6 = ([(0, 0, 0), (0, 0, 0), (1, 0, 0), (-1, 0, 0)],
          [(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0)],
          [(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0)])
    M7 = ([(0, 0, 0), (1, 0, 0), (0, 0, 0), (-1, 0, 0)],
          [(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
          [(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 0, 0)])
    macros = [M1, M2, M3, M4, M5, M6, M7]

    U = [[0] * 53 for _ in range(n * m)]
    V = [[0] * 53 for _ in range(m * p)]
    W = [[0] * 53 for _ in range(p * n)]

    col = 0
    # 4 column blocks of B (each block covers 2 B columns: 0,1 / 2,3)
    for jblk in range(2):
        # 2 row blocks of C (each covers 2 C rows: 0,1 / 2,3)
        for iblk in range(2):
            for u, v, w in macros:
                for a in range(2):
                    for b in range(2):
                        if u[a][b]:
                            U[(2 * iblk + a) * m + (2 * jblk + b)][col] = u[a][b]
                        if v[a][b]:
                            V[(2 * jblk + a) * p + (2 * iblk + b)][col] = v[a][b]
                        if w[a][b]:
                            W[(2 * iblk + a) * n + (2 * jblk + b)][col] = w[a][b]
                col += 1

    # 4 schoolbook terms for the 5th column of B (index 4), C columns 0..3
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
