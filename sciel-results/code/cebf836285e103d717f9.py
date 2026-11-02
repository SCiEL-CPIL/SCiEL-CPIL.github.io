"""Matrix multiplication 4,4,5 — block-strassen rank-64 decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    rank = 64
    U = [[0] * rank for _ in range(n * m)]   # 16 x 64
    V = [[0] * rank for _ in range(m * p)]   # 20 x 64
    W = [[0] * rank for _ in range(p * n)]   # 20 x 64

    # 2x2x2 Strassen pattern (indices local 0..3)
    strassen = [
        ({0: 1, 1: 1, 2: 1, 3: 1}, {0: 1, 2: 1, 4: 1, 6: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),
        ({1: 1, 3: 1},              {0: -1, 2: 1, 4: -1, 6: 1}, {2: 1, 3: 1}),
        ({0: 1, 2: 1},              {1: 1, 3: -1, 5: 1, 7: -1}, {0: 1, 1: 1}),
        ({3: 1},                    {1: 1, 5: 1},               {2: -1, 3: 1}),
        ({0: 1},                    {1: 1, 3: 1, 5: 1, 7: 1},   {2: 1, 3: 1}),
        ({2: 1},                    {0: 1, 4: 1},               {0: -1, 1: 1}),
        ({0: 1, 2: 1},              {0: 1, 1: 1, 2: 1, 3: 1},   {2: -1, 3: -1}),
    ]

    # 2x2x3 rank-4 pattern
    # U local: 2x2 indices 0..3; V local: 2x3 indices 0..5; W local: 3x2 indices 0..5
    rank4_223 = [
        ({0: 1},              {0: 1, 1: 1, 2: 1}, {0: 1, 2: 1, 4: 1}),
        ({1: 1},              {3: 1, 4: 1, 5: 1}, {0: 1, 2: 1, 4: 1}),
        ({2: 1},              {0: 1, 1: 1, 2: 1}, {1: 1, 3: 1, 5: 1}),
        ({3: 1},              {3: 1, 4: 1, 5: 1}, {1: 1, 3: 1, 5: 1}),
    ]

    def add_strassen(i0, j0, k0, col_start):
        """A block (i0..i0+1, j0..j0+1) * B block (j0..j0+1, k0..k0+1) -> C block (k0..k0+1, i0..i0+1)."""
        for t, (uc, vc, wc) in enumerate(strassen):
            col = col_start + t
            for u_idx, val in uc.items():
                li, lj = u_idx // 2, u_idx % 2
                U[(i0 + li) * m + (j0 + lj)][col] += val
            for v_idx, val in vc.items():
                lj, lk = v_idx // 2, v_idx % 2
                V[(j0 + lj) * p + (k0 + lk)][col] += val
            for w_idx, val in wc.items():
                lk, li = w_idx // 2, w_idx % 2
                W[(k0 + lk) * n + (i0 + li)][col] += val

    def add_rank4(i0, j0, k0, col_start):
        """A block (i0..i0+1, j0..j0+1) * B block (j0..j0+1, k0..k0+2) -> C block (k0..k0+2, i0..i0+1)."""
        for t, (uc, vc, wc) in enumerate(rank4_223):
            col = col_start + t
            for u_idx, val in uc.items():
                li, lj = u_idx // 2, u_idx % 2
                U[(i0 + li) * m + (j0 + lj)][col] += val
            for v_idx, val in vc.items():
                lj, lk = v_idx // 3, v_idx % 3
                V[(j0 + lj) * p + (k0 + lk)][col] += val
            for w_idx, val in wc.items():
                lk, li = w_idx // 2, w_idx % 2
                W[(k0 + lk) * n + (i0 + li)][col] += val

    col = 0

    # C11 (k=0,1; i=0,1) = A11*B11 + A12*B21
    add_strassen(0, 0, 0, col); col += 7   # A11*B11
    add_strassen(0, 2, 0, col); col += 7   # A12*B21

    # C12 (k=0,1; i=2,3) = A21*B11 + A22*B21
    add_strassen(2, 0, 0, col); col += 7   # A21*B11
    add_strassen(2, 2, 0, col); col += 7   # A22*B21

    # C21 (k=2,3,4; i=0,1) = A11*B12 + A12*B22
    add_rank4(0, 0, 2, col); col += 4      # A11*B12
    add_rank4(0, 2, 2, col); col += 4      # A12*B22

    # C22 (k=2,3,4; i=2,3) = A21*B12 + A22*B22
    add_rank4(2, 0, 2, col); col += 4      # A21*B12
    add_rank4(2, 2, 2, col); col += 4      # A22*B22

    # Verify column count
    assert col == rank, f"Expected {rank} columns, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
