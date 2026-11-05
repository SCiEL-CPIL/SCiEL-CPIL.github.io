"""Matrix multiplication 6x6x6 via 2x2 Strassen block decomposition
with each 3x3 block product done schoolbook (rank 27).
Total rank: 7 * 27 = 189.

The Strassen decomposition at the 2x2 block level:
  P1 = (A00+A11)(B00+B11)
  P2 = (A10+A11)B00
  P3 = A00(B01-B11)
  P4 = A11(B10-B00)
  P5 = (A00+A01)B11
  P6 = (A10-A00)(B00+B01)
  P7 = (A01-A11)(B10+B11)

Output reconstruction:
  C00 = P1 + P4 - P5 + P7
  C01 = P3 + P5
  C10 = P2 + P4
  C11 = P1 - P2 + P3 + P6
"""

import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size

    # Strassen 7 virtual products at the 2x2 block level.
    # Each term: (a_comb, b_comb) where a_comb is list of (block_idx, coeff)
    # for A, b_comb is list of (block_idx, coeff) for B.
    # Block indices: 0=TL(0,0), 1=TR(0,1), 2=BL(1,0), 3=BR(1,1)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1 = (A00+A11)(B00+B11)
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2 = (A10+A11)B00
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3 = A00(B01-B11)
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4 = A11(B10-B00)
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5 = (A00+A01)B11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6 = (A10-A00)(B00+B01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7 = (A01-A11)(B10+B11)
    ]

    # Output reconstruction: for each Strassen term P_t, which output blocks
    # get which coefficient.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # output_contrib[t] = list of (block_idx, coeff)
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]

    rank = 7 * 27  # 189

    U = [[0] * rank for _ in range(n * m)]  # shape (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # shape (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # shape (36, 189)

    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for i in range(bs):       # local row in A block (and C block)
            for k in range(bs):   # local col in A block / local row in B block
                for j in range(bs):  # local col in B block (and C block)
                    # U: contributions from A entries
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * bs
                        col_a = (blk_a % 2) * bs
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx][col] += coeff_a

                    # V: contributions from B entries
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * bs
                        col_b = (blk_b % 2) * bs
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx][col] += coeff_b

                    # W: contributions to C entries
                    # W index convention: W[k*n+i, r] where k is column of C, i is row of C
                    # So w_idx = c_col * n + c_row
                    for (blk_c, coeff_c) in output_contrib[term_idx]:
                        row_c = (blk_c // 2) * bs
                        col_c = (blk_c % 2) * bs
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx][col] += coeff_c

                    col += 1

    return {"U": U, "V": V, "W": W}
