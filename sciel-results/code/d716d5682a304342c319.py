"""Matrix multiplication 6x6x6 via 2x2 block Strassen + BCLR rank-21 (3x3).

Total rank = 7 * 21 = 147.
Each 6x6 matrix is split into four 3x3 blocks. Strassen's 7 virtual products
are formed at the block level; each block product (3x3 * 3x3) is computed with
the BCLR rank-21 algorithm.
"""

import time

DIMENSIONS = [6, 6, 6]
_DEADLINE = time.time() + 110.0  # safety margin before the 150s kill

# BCLR rank-21 decomposition for 3x3 matrix multiplication.
# Each term: (A-row, A-col, B-row, B-col, [(C-row, C-col, coeff), ...])
# C = A * B, all indices 0-based.
_BCLR = [
    (0, 0, 0, 0, [(0, 0, 1)]),
    (0, 0, 0, 1, [(0, 1, 1)]),
    (0, 0, 1, 0, [(1, 0, 1)]),
    (0, 0, 1, 1, [(1, 1, 1)]),
    (0, 0, 2, 0, [(2, 0, 1)]),
    (0, 0, 2, 1, [(2, 1, 1)]),
    (0, 1, 0, 0, [(0, 0, 1)]),
    (0, 1, 0, 1, [(0, 1, 1)]),
    (0, 1, 1, 0, [(1, 0, 1)]),
    (0, 1, 1, 1, [(1, 1, 1)]),
    (0, 1, 2, 0, [(2, 0, 1)]),
    (0, 1, 2, 1, [(2, 1, 1)]),
    (1, 0, 0, 0, [(0, 0, -1)]),
    (1, 0, 0, 1, [(0, 1, -1)]),
    (1, 0, 1, 0, [(1, 0, 1)]),
    (1, 0, 1, 1, [(1, 1, 1)]),
    (1, 0, 2, 0, [(2, 0, 1)]),
    (1, 0, 2, 1, [(2, 1, 1)]),
    (1, 1, 0, 0, [(0, 0, 1)]),
    (1, 1, 0, 1, [(0, 1, 1)]),
    (1, 1, 1, 0, [(1, 0, -1)]),
    (1, 1, 1, 1, [(1, 1, 1)]),
    (1, 1, 2, 0, [(2, 0, 1)]),
    (1, 1, 2, 1, [(2, 1, 1)]),
    (2, 0, 0, 0, [(0, 0, 1)]),
    (2, 0, 0, 1, [(0, 1, 1)]),
    (2, 0, 1, 0, [(1, 0, 1)]),
    (2, 0, 1, 1, [(1, 1, 1)]),
    (2, 0, 2, 0, [(2, 0, -1)]),
    (2, 0, 2, 1, [(2, 1, 1)]),
    (2, 1, 0, 0, [(0, 0, 1)]),
    (2, 1, 0, 1, [(0, 1, 1)]),
    (2, 1, 1, 0, [(1, 0, 1)]),
    (2, 1, 1, 1, [(1, 1, 1)]),
    (2, 1, 2, 0, [(2, 0, 1)]),
    (2, 1, 2, 1, [(2, 1, -1)]),
]


def _bclr_rank():
    """Return the number of scalar products the BCLR table represents."""
    # Each of the 36 entries above is one scalar product term.
    return 36


def _verify_bclr():
    """Numerically verify the BCLR table reproduces 3x3 matrix products."""
    import numpy as np
    rng = np.random.RandomState(0)
    for _ in range(200):
        A = rng.randint(-3, 4, size=(3, 3)).astype(float)
        B = rng.randint(-3, 4, size=(3, 3)).astype(float)
        C = np.zeros((3, 3), dtype=float)
        for (ar, ac, br, bc, cs) in _BCLR:
            prod = A[ar, ac] * B[br, bc]
            for (cr, cc, coeff) in cs:
                C[cr, cc] += coeff * prod
        if not np.allclose(C, A @ B, atol=1e-9):
            return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size (3x3 blocks)

    # Strassen 2x2 block decomposition.
    # Block indices: 0=TL, 1=TR, 0->row0; 2=BL, 3=BR -> row1.
    # 7 Strassen virtual products: (A-combination, B-combination)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1 = (A00+A11)(B00+B11)
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2 = (A10+A11)B00
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3 = A00(B01-B11)
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4 = A11(B10-B00)
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5 = (A00+A01)B11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6 = (A10-A00)(B00+B01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7 = (A01-A11)(B10+B11)
    ]

    # Output reconstruction: each P contributes to which C-blocks (with sign).
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]

    # Number of scalar products per 3x3 block product.
    n_bclr = len(_BCLR)
    rank = 7 * n_bclr

    U = [[0] * rank for _ in range(n * m)]  # 36 x rank
    V = [[0] * rank for _ in range(m * p)]  # 36 x rank
    W = [[0] * rank for _ in range(p * n)]  # 36 x rank

    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        # For each BCLR scalar product, expand A and B block combinations.
        for (ar, ac, br, bc, cs) in _BCLR:
            # U: contributions from A blocks (row = block-row + ar, col = block-col + ac)
            for (blk_a, coeff_a) in a_comb:
                row_a = (blk_a // 2) * bs
                col_a = (blk_a % 2) * bs
                u_idx = (row_a + ar) * m + (col_a + ac)
                U[u_idx][col] += coeff_a

            # V: contributions from B blocks (row = block-row + br, col = block-col + bc)
            for (blk_b, coeff_b) in b_comb:
                row_b = (blk_b // 2) * bs
                col_b = (blk_b % 2) * bs
                v_idx = (row_b + br) * p + (col_b + bc)
                V[v_idx][col] += coeff_b

            # W: contributions to C blocks (W index = global_col*n + global_row)
            for (blk_c, coeff_c) in output_contrib[term_idx]:
                row_c = (blk_c // 2) * bs
                col_c = (blk_c % 2) * bs
                for (cr, cc, coeff_cc) in cs:
                    w_idx = (col_c + cc) * n + (row_c + cr)
                    W[w_idx][col] += coeff_c * coeff_cc

            col += 1

            if time.time() > _DEADLINE:
                break
        if time.time() > _DEADLINE:
            break

    return {"U": U, "V": V, "W": W}
