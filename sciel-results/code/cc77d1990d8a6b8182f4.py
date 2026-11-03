"""Strassen (2x2 blocks) x schoolbook (3x3 blocks) exact tensor decomposition for 6x6 matmul."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120.0
    n = m = p = 6
    B = 3  # block size

    # Strassen 2x2 block structure.
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # 7 products P_j = (lincomb A)(lincomb B); C blocks are lincombs of P_j.
    # (s, t) -> (a, b) linear combination, sign in {+1, -1}
    products = [
        (0, 0, +1), (0, 1, +1), (1, 0, +1), (1, 1, +1),                       # P0 = (A00+A11)(B00+B11)
        (1, 0, +1), (1, 1, +1), (0, 0, +1), (0, 1, +1),                        # P1 = (A10+A11)B00
        (0, 0, +1), (1, 1, -1), (0, 1, +1), (1, 0, -1),                        # P2 = A00(B01-B11)
        (0, 1, +1), (1, 0, -1), (1, 1, +1), (0, 0, -1),                        # P3 = A11(B10-B00)
        (0, 0, +1), (1, 1, +1), (0, 0, +1), (1, 1, +1),                        # P4 = (A00+A01)B11
        (1, 0, +1), (0, 0, -1), (1, 0, +1), (0, 1, -1),                        # P5 = (A10-A00)(B00+B01)
        (0, 1, +1), (1,1, -1), (0, 1, +1), (1, 0, -1),                         # P6 = (A01-A11)(B10+B11)
    ]

    # C block (i_block, k_block) = sum_j coef[j] * P_j
    c_combos = {
        (0, 0): [(0, +1), (2, +1), (4, +1), (5, +1)],
        (0, 1): [(2, +1), (4, -1)],
        (1, 0): [(1, +1), (3, +1)],
        (1, 1): [(0, +1), (3, +1), (5, -1), (6, +1)],
    }

    rank = 7 * 9  # 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for j in range(7):
        a_terms = products[j][0:4]   # (a_block, b_block, sign)
        b_terms = products[j][4:8]
        for a in range(B):
            for b in range(B):
                col = j * 9 + a * B + b
                # U factor: a_block row offset, b_block col offset
                for (ab, bb, sa) in a_terms:
                    U[ab * B + a][bb * B + b] += sa
                # V factor: a_block row offset, b_block col offset
                for (ab, bb, sb) in b_terms:
                    V[ab * B + a][bb * B + b] += sb
                # W factor: output block C[ib, kb] = sum coef*P_j, term a (row a) * b (col b)
                for (ib, kb) in c_combos:
                    for (jj, coef) in c_combos[(ib, kb)]:
                        if jj == j:
                            W[kb * B + b][ib * B + a] += coef

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
