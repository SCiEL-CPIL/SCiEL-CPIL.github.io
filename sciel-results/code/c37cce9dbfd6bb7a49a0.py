"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition:
    # Block A: j in {0,1}, k in {0,1,2} -> 2x2x3, rank 12
    # Block B: j in {2,3,4}, k in {0,1,2} -> 2x3x3, rank 18
    # Block C: j in {0,1}, k in {3,4,5} -> 2x2x3, rank 12
    # Block D: j in {2,3,4}, k in {3,4,5} -> 2x3x3, rank 18
    # Total rank = 12 + 18 + 12 + 18 = 60

    rank = 60

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen terms for 2x2x2:
    # Each term is (u_terms, v_terms, w_terms) where each term is (row, col, val)
    # u: (li, lj, val) -> U[li*m + lj]
    # v: (lj, lk, val) -> V[lj*p + lk]
    # w: (li, lk, val) -> W[lk*n + li]
    strassen_2x2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],           [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],           [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)],           [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)],   [(1,1,1)],           [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)],  [(0,0,1),(0,1,1)],   [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],  [(1,0,1),(1,1,1)],   [(0,0,1)]),
    ]

    def apply_strassen_block(i_base, j_base, k_base):
        """Apply 7 Strassen terms for a 2x2x2 block starting at (i_base, j_base, k_base).
        Returns new column index."""
        nonlocal col
        for u_terms, v_terms, w_terms in strassen_2x2x2:
            for (li, lj, val) in u_terms:
                U[(i_base + li) * m + (j_base + lj)][col] += val
            for (lj, lk, val) in v_terms:
                V[(j_base + lj) * p + (k_base + lk)][col] += val
            for (li, lk, val) in w_terms:
                W[(k_base + lk) * n + (i_base + li)][col] += val
            col += 1
        return col

    def apply_schoolbook_block(i_base, j_base, k_base, ni, nj, nk):
        """Apply schoolbook multiplication for ni x nj x nk block.
        Returns new column index."""
        nonlocal col
        for li in range(ni):
            for lj in range(nj):
                for lk in range(nk):
                    U[(i_base + li) * m + (j_base + lj)][col] = 1
                    V[(j_base + lj) * p + (k_base + lk)][col] = 1
                    W[(k_base + lk) * n + (i_base + li)][col] = 1
                    col += 1
        return col

    # Block A: i in {0,1}, j in {0,1}, k in {0,1,2} -> 2x2x3
    # Split into 2x2x2 (k=0,1) + 2x1x2 (k=2)
    # 2x2x2 part: rank 7
    col = apply_strassen_block(0, 0, 0)
    # 2x1x2 part (j=0, k=2,3): schoolbook, rank 4
    col = apply_schoolbook_block(0, 0, 2, 2, 1, 2)

    # Block B: i in {0,1}, j in {2,3,4}, k in {0,1,2} -> 2x3x3
    # j=2: 2x1x3 schoolbook, rank 6
    col = apply_schoolbook_block(0, 2, 0, 2, 1, 3)
    # j in {3,4}: 2x2x3, split into 2x2x2 + 2x1x2
    col = apply_strassen_block(0, 3, 0)
    col = apply_schoolbook_block(0, 3, 2, 2, 1, 2)

    # Block C: i in {0,1}, j in {0,1}, k in {3,4,5} -> 2x2x3
    # 2x2x2 (k=3,4) + 2x1x2 (k=5)
    col = apply_strassen_block(0, 0, 3)
    col = apply_schoolbook_block(0, 0, 5, 2, 1, 2)

    # Block D: i in {0,1}, j in {2,3,4}, k in {3,4,5} -> 2x3x3
    # j=2: 2x1x3 schoolbook, rank 6
    col = apply_schoolbook_block(0, 2, 3, 2, 1, 3)
    # j in {3,4}: 2x2x3, split into 2x2x2 + 2x1x2
    col = apply_strassen_block(0, 3, 3)
    col = apply_schoolbook_block(0, 3, 5, 2, 1, 2)

    assert col == rank, f"Column mismatch: {col} vs {rank}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
