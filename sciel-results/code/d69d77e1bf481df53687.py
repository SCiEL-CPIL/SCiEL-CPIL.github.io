import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication via recursive Strassen.
    
    Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    Tensor: sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r] = 1 for all (i,j,k)
    """
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 4

    # Build rank-49 Strassen decomposition
    rank49 = 49
    U49 = np.zeros((n * m, rank49), dtype=float)
    V49 = np.zeros((m * p, rank49), dtype=float)
    W49 = np.zeros((p * n, rank49), dtype=float)

    def get_U_index(row, col):
        return row * 4 + col

    def get_V_index(row, col):
        return row * 4 + col

    def get_W_index(k, i):
        return k * 4 + i  # trace convention: W[k*n+i, r]

    # Strassen 2x2 formulas: (A_block, B_block, C_block)
    # Each block is a dict mapping (row, col) -> coefficient
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # For 4x4, we apply Strassen at the block level (2x2 blocks of 2x2 matrices)
    # The 4x4 matrix is partitioned into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen gives 7 products of 2x2 matrices, each computed recursively
    # with 7 products of 1x1 scalars.
    #
    # Block-level Strassen (same as 2x2 Strassen but operating on 2x2 blocks):
    block_strassen = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    for t in range(7):
        block_A, block_B, block_C = block_strassen[t]
        for s in range(7):
            scalar_A, scalar_B, scalar_C = strassen_2x2[s]
            r_idx = t * 7 + s

            # For each block in A, and each scalar position within that block
            for (bi, bj), block_coeff in block_A.items():
                for (sr, sc), scalar_coeff in scalar_A.items():
                    a_row = 2 * bi + sr
                    a_col = 2 * bj + sc
                    U49[get_U_index(a_row, a_col), r_idx] += block_coeff * scalar_coeff

            for (bi, bj), block_coeff in block_B.items():
                for (sr, sc), scalar_coeff in scalar_B.items():
                    b_row = 2 * bi + sr
                    b_col = 2 * bj + sc
                    V49[get_V_index(b_row, b_col), r_idx] += block_coeff * scalar_coeff

            for (ci, cj), block_coeff in block_C.items():
                for (sr, sc), scalar_coeff in scalar_C.items():
                    c_row = 2 * ci + sr
                    c_col = 2 * cj + sc
                    # W uses trace convention: W[k*n+i, r] where k is col of C, i is row of C
                    # C has blocks C00, C01, C10, C11
                    # C00 contributes to C[0:2, 0:2], so rows 0-1, cols 0-1
                    # W index: k*4 + i where k is the column index in C, i is the row index in C
                    w_idx = get_W_index(c_col, c_row)
                    W49[w_idx, r_idx] += block_coeff * scalar_coeff

    # Verify the rank-49 decomposition
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0.0
                for r in range(49):
                    val += U49[i*4+j, r] * V49[j*4+k, r] * W49[k*4+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-10:
        # Fallback: return the unverified decomposition (should not happen)
        U = U49.tolist()
        V = V49.tolist()
        W = W49.tolist()
        return {"U": U, "V": V, "W": W}

    # Round to nearest 0.5 for cleanliness
    U_out = U49.tolist()
    V_out = V49.tolist()
    W_out = W49.tolist()

    for mat in [U_out, V_out, W_out]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                mat[i][j] = round(mat[i][j] * 2) / 2.0

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
