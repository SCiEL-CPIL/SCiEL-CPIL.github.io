"""Block-Strassen decomposition for 6x6 matrix multiplication.

Strategy:
  - Partition the 6x6 matrices into 2x2 blocks of 3x3 sub-matrices.
  - Apply Strassen's 7-multiplication scheme at the block level.
  - Each of the 7 intermediate products is a 3x3 x 3x3 multiplication.
  - Each 3x3 multiplication uses the schoolbook rank-27 decomposition.
  - Total rank = 7 * 27 = 189.

The evaluator convention: W[k*n+i,r] for the output factor.
"""
import time

DIMENSIONS = [6, 6, 6]


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size
    nb = 2  # number of blocks along each dimension

    # Strassen coefficients for 2x2 block multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Define Strassen terms: (a_blocks, b_blocks, c_blocks)
    # Each is a dict: {(block_i, block_j): coefficient}
    strassen_defs = [
        # M1 = (A11+A22)(B11+B22) -> C11 + C22
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        # M2 = (A21+A22)B11 -> C21 - C22
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        # M3 = A11(B12-B22) -> C12 + C22
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        # M4 = A22(B21-B11) -> C11 + C21
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        # M5 = (A11+A12)B22 -> -C11 + C12
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        # M6 = (A21-A11)(B11+B12) -> C22
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        # M7 = (A12-A22)(B21+B22) -> C11
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]

    total_rank = 7 * 27  # 189

    U = [[0] * total_rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 189

    col = 0
    for mi, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_defs):
        for ia in range(bs):
            for ja in range(bs):
                for kb in range(bs):
                    # For each sub-index (ia, ja, kb) in the 3x3 block multiplication:
                    # A_sub[ia, ja] * B_sub[ja, kb] -> C_sub[kb, ia]
                    
                    # Set U entries for all A-blocks involved
                    for (bi, bj), coeff_a in a_blocks.items():
                        row_u = (bi * bs + ia) * m + (bj * bs + ja)
                        U[row_u][col] += coeff_a
                    
                    # Set V entries for all B-blocks involved
                    for (bj, bk), coeff_b in b_blocks.items():
                        row_v = (bj * bs + ja) * p + (bk * bs + kb)
                        V[row_v][col] += coeff_b
                    
                    # Set W entries for all C-blocks involved
                    for (bk, bi), coeff_c in c_blocks.items():
                        row_w = (bk * bs + kb) * n + (bi * bs + ia)
                        W[row_w][col] += coeff_c
                    
                    col += 1

    return {"U": U, "V": V, "W": W}
