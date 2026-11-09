import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # evaluator kills at 150s; keep a safety margin

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5

    # ---- Strassen decomposition for [2^k, 2^k, 2^k] -> rank 7^k ----
    # Returns lists of (u_idx, v_idx, w_idx, coeff).
    # u_idx = (i, j) for A, v_idx = (j, k) for B, w_idx = (k, i) for C.
    def strassen_terms(k):
        if k == 0:
            return [((0, 0), (0, 0), (0, 0), 1)]
        sub = strassen_terms(k - 1)
        s = 2 ** (k - 1)

        # Block offset maps: (block_row, block_col) -> (row_offset, col_offset)
        Amap = {(0, 0): (0, 0), (0, 1): (0, s), (1, 0): (s, 0), (1, 1): (s, s)}
        Bmap = Amap
        Cmap = Amap

        # Each of the 7 products: (list of A blocks, list of B blocks, coeff)
        products = [
            ([(0, 0), (1, 1)], [(0, 0), (1, 1)], 1),
            ([(1, 0), (1, 1)], [(0, 0)], 1),
            ([(0, 0)], [(0, 1), (1, 1)], 1),
            ([(1, 1)], [(1, 0), (0, 0)], 1),
            ([(0, 0), (0, 1)], [(1, 1)], 1),
            ([(1, 0), (0, 0)], [(0, 0), (0, 1)], 1),
            ([(0, 1), (1, 1)], [(1, 0), (1, 1)], 1),
        ]
        # C block -> list of (product_index, sign)
        Ccomb = {
            (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
            (0, 1): [(2, 1), (4, 1)],
            (1, 0): [(1, 1), (3, 1)],
            (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)],
        }

        def shift(idx, off):
            return (idx[0] + off[0], idx[1] + off[1])

        terms = []
        for cblock, comps in Ccomb.items():
            croff = Cmap[cblock]
            for pi, sign in comps:
                a_blocks, b_blocks, pcoeff = products[pi]
                for (ar, ac), (br, bc) in [(a, b) for a in a_blocks for b in b_blocks]:
                    aoff = Amap[(ar, ac)]
                    boff = Bmap[(br, bc)]
                    for (ui, uj), (vi, vj), (wi, wj), c in sub:
                        coeff = sign * pcoeff * c
                        terms.append((
                            shift((ui, uj), aoff),
                            shift((vi, vj), boff),
                            shift((wi, wj), croff),
                            coeff,
                        ))
        return terms

    # ---- Assemble the full [4,5,5] decomposition ----
    # Block split of the 5x5 matrices:
    #   A: rows 0..3 (n=4), cols 0..3 (b) and col 4 (c)
    #   B: rows 0..3 (b) and row 4 (c), cols 0..3 (b) and col 4 (d)
    #   C: rows 0..3 (b) and row 4 (d), cols 0..3 (b) and col 4 (d)
    #
    # Part 1: [4,4,4] on (b,b,b) via Strassen -> rank 49
    # Part 2: [4,1,4] on (b,c,b) via schoolbook -> rank 16
    # Part 3: [4,5,1] on (b,(b,c),d) via schoolbook -> rank 20
    # Total rank 85.

    s49 = strassen_terms(2)  # 49 terms for 4x4x4

    # Indexing conventions:
    #   U index = i*m + j  (i in 0..n-1, j in 0..m-1)
    #   V index = j*p + k  (j in 0..m-1, k in 0..p-1)
    #   W index = k*n + i  (k in 0..p-1, i in 0..n-1)
    rank = 49 + 16 + 20
    U = [[0] * rank for _ in range(n * m)]  # 20
    V = [[0] * rank for _ in range(m * p)]  # 25
    W = [[0] * rank for _ in range(p * n)]  # 20

    col = 0
    # Part 1: [4,4,4] Strassen, b in 0..3
    for (ui, uj), (vi, vj), (wi, wj), c in s49:
        U[ui * 5 + uj][col] = c
        V[vi * 5 + vj][col] = c
        W[wi * 4 + wj][col] = c
        col += 1

    # Part 2: [4,1,4] schoolbook, A col = 4, B row = 4
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1

    # Part 3: [4,5,1] schoolbook, B col = 4, C col = 4
    for i in range(4):
        for j in range(5):
            U[i * 5 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    if col != rank:
        # Fallback: full schoolbook (rank 100)
        rank = 100
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    U[i * 5 + j][col] = 1
                    V[j * 5 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
