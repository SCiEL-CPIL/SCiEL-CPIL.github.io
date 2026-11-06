"""Matrix multiplication 6x6x6 via recursive Strassen decomposition.
Level 1: 2x2 blocks of 3x3, Strassen 7 products.
Level 2: each 3x3 product decomposed into 3x3 schoolbook (rank 27).
Total rank: 7 * 27 = 189.
"""

import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin within 120s limit

    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size

    # Strassen 7 virtual products at the 2x2 block level
    # Block layout: 0=TL, 1=TR, 2=BL, 3=BR
    # Block (bi): row offset = (bi // 2) * bs, col offset = (bi % 2) * bs

    # Each term: (list of (block_idx, coeff) for A, list of (block_idx, coeff) for B)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # P1 = (A00+A11)(B00+B11)
        ([(2, 1), (3, 1)], [(0, 1)]),            # P2 = (A10+A11)B00
        ([(0, 1)], [(1, 1), (3, -1)]),           # P3 = A00(B01-B11)
        ([(3, 1)], [(2, 1), (0, -1)]),           # P4 = A11(B10-B00)
        ([(0, 1), (1, 1)], [(3, 1)]),            # P5 = (A00+A01)B11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # P6 = (A10-A00)(B00+B01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # P7 = (A01-A11)(B10+B11)
    ]

    # Output reconstruction:
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

    rank = 7 * bs * bs * bs  # 7 * 27 = 189

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        for i in range(bs):   # local row in 3x3 block
            for k in range(bs):  # local col in A / local row in B
                for j in range(bs):  # local col in B / local col in C
                    # U contributions: A entries
                    for (blk_a, coeff_a) in a_comb:
                        row_a = (blk_a // 2) * bs
                        col_a = (blk_a % 2) * bs
                        u_idx = (row_a + i) * m + (col_a + k)
                        U[u_idx, col] += coeff_a

                    # V contributions: B entries
                    for (blk_b, coeff_b) in b_comb:
                        row_b = (blk_b // 2) * bs
                        col_b = (blk_b % 2) * bs
                        v_idx = (row_b + k) * p + (col_b + j)
                        V[v_idx, col] += coeff_b

                    # W contributions: C entries (W index = global_col * n + global_row)
                    for (blk_c, coeff_c) in output_contrib[term_idx]:
                        row_c = (blk_c // 2) * bs
                        col_c = (blk_c % 2) * bs
                        w_idx = (col_c + j) * n + (row_c + i)
                        W[w_idx, col] += coeff_c

                    col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Quick verification
    if time.time() < deadline - 5:
        U_np = U
        V_np = V
        W_np = W
        # Verify tensor: sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] == 1 for all i,k,j
        # Check a representative subset
        for test_i in range(6):
            for test_k in range(6):
                for test_j in range(6):
                    coeff = 0
                    for r in range(rank):
                        u_val = U_np[test_i * m + test_k, r]
                        if u_val == 0:
                            continue
                        v_val = V_np[test_k * p + test_j, r]
                        if v_val == 0:
                            continue
                        w_val = W_np[test_j * n + test_i, r]
                        if w_val == 0:
                            continue
                        coeff += u_val * v_val * w_val
                    if coeff != 1:
                        raise ValueError(f"Tensor verification failed at ({test_i},{test_k},{test_j}): got {coeff}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
