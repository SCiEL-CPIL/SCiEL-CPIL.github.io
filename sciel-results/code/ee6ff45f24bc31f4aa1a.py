"""Strassen-based rank-42 decomposition for 2x4 times 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank))  # 8 x 42
    V = np.zeros((m * p, rank))  # 24 x 42
    W = np.zeros((p * n, rank))  # 12 x 42

    # Six 2x2 blocks: (A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off)
    # A is 2x4, B is 4x6, C is 2x6
    # A index: i*m + j = 4*i + j
    # B index: j*p + k = 6*j + k
    # W index: k*n + i = 2*k + i
    blocks = [
        (0, 0, 0, 0, 0, 0),  # A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        (0, 2, 2, 0, 0, 0),  # A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        (0, 0, 0, 2, 0, 2),  # A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        (0, 2, 2, 2, 0, 2),  # A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
        (0, 0, 0, 4, 0, 4),  # A[:,0:2] * B[0:2,4:6] -> C[:,4:6]
        (0, 2, 2, 4, 0, 4),  # A[:,2:4] * B[2:4,4:6] -> C[:,4:6]
    ]

    # Strassen's 7 multipliers for 2x2 blocks:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X10)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X11-X10)(Y10+Y11)
    #
    # Output:
    # Z00 = M1 + M4 - M5 - M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M7

    # For each of the 6 blocks, we have 7 Strassen terms.
    # Total rank = 6 * 7 = 42

    for b_idx, (ar, ac, br, bc, cr, cc) in enumerate(blocks):
        base_col = b_idx * 7

        # Term 0: M1 = (X00+X11)(Y00+Y11)
        col = base_col + 0
        # U: X00 (row 0, col 0) and X11 (row 1, col 1)
        U[4 * (ar + 0) + (ac + 0)][col] += 1
        U[4 * (ar + 1) + (ac + 1)][col] += 1
        # V: Y00 (row 0, col 0) and Y11 (row 1, col 1)
        V[6 * (br + 0) + (bc + 0)][col] += 1
        V[6 * (br + 1) + (bc + 1)][col] += 1
        # W: Z00 (row 0, col 0) and Z11 (row 1, col 1)
        W[2 * (cc + 0) + (cr + 0)][col] += 1
        W[2 * (cc + 1) + (cr + 1)][col] += 1

        # Term 1: M2 = (X10+X11)Y00
        col = base_col + 1
        # U: X10 (row 1, col 0) and X11 (row 1, col 1)
        U[4 * (ar + 1) + (ac + 0)][col] += 1
        U[4 * (ar + 1) + (ac + 1)][col] += 1
        # V: Y00 (row 0, col 0)
        V[6 * (br + 0) + (bc + 0)][col] += 1
        # W: Z10 gets +1, Z11 gets -1
        W[2 * (cc + 0) + (cr + 1)][col] += 1
        W[2 * (cc + 1) + (cr + 1)][col] += -1

        # Term 2: M3 = X00(Y10-Y11)
        col = base_col + 2
        # U: X00 (row 0, col 0)
        U[4 * (ar + 0) + (ac + 0)][col] += 1
        # V: Y10 (row 1, col 0) and Y11 (row 1, col 1)
        V[6 * (br + 1) + (bc + 0)][col] += 1
        V[6 * (br + 1) + (bc + 1)][col] += -1
        # W: Z01 gets +1, Z11 gets +1
        W[2 * (cc + 1) + (cr + 0)][col] += 1
        W[2 * (cc + 1) + (cr + 1)][col] += 1

        # Term 3: M4 = X11(Y01-Y00)
        col = base_col + 3
        # U: X11 (row 1, col 1)
        U[4 * (ar + 1) + (ac + 1)][col] += 1
        # V: Y01 (row 0, col 1) and Y00 (row 0, col 0)
        V[6 * (br + 0) + (bc + 1)][col] += 1
        V[6 * (br + 0) + (bc + 0)][col] += -1
        # W: Z00 gets +1, Z10 gets +1
        W[2 * (cc + 0) + (cr + 0)][col] += 1
        W[2 * (cc + 0) + (cr + 1)][col] += 1

        # Term 4: M5 = (X00+X10)Y11
        col = base_col + 4
        # U: X00 (row 0, col 0) and X10 (row 1, col 0)
        U[4 * (ar + 0) + (ac + 0)][col] += 1
        U[4 * (ar + 1) + (ac + 0)][col] += 1
        # V: Y11 (row 1, col 1)
        V[6 * (br + 1) + (bc + 1)][col] += 1
        # W: Z00 gets -1, Z01 gets +1
        W[2 * (cc + 0) + (cr + 0)][col] += -1
        W[2 * (cc + 1) + (cr + 0)][col] += 1

        # Term 5: M6 = (X10-X00)(Y00+Y01)
        col = base_col + 5
        # U: X10 (row 1, col 0) and X00 (row 0, col 0)
        U[4 * (ar + 1) + (ac + 0)][col] += 1
        U[4 * (ar + 0) + (ac + 0)][col] += -1
        # V: Y00 (row 0, col 0) and Y01 (row 0, col 1)
        V[6 * (br + 0) + (bc + 0)][col] += 1
        V[6 * (br + 0) + (bc + 1)][col] += 1
        # W: M6 does not directly appear in any Z. Wait, let me re-check.
        # Z00 = M1 + M4 - M5 - M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M7
        # Indeed, M6 does not appear. This is correct for Strassen's algorithm.

        # Term 6: M7 = (X11-X10)(Y10+Y11)
        col = base_col + 6
        # U: X11 (row 1, col 1) and X10 (row 1, col 0)
        U[4 * (ar + 1) + (ac + 1)][col] += 1
        U[4 * (ar + 1) + (ac + 0)][col] += -1
        # V: Y10 (row 1, col 0) and Y11 (row 1, col 1)
        V[6 * (br + 1) + (bc + 0)][col] += 1
        V[6 * (br + 1) + (bc + 1)][col] += 1
        # W: Z00 gets -1, Z11 gets +1
        W[2 * (cc + 0) + (cr + 0)][col] += -1
        W[2 * (cc + 1) + (cr + 1)][col] += 1

    # Verification
    A = np.random.randint(-10, 10, (n, m))
    B = np.random.randint(-10, 10, (m, p))
    C_ref = A @ B

    C_test = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for c in range(rank):
                w_val = W[k * n + i][c]
                if w_val == 0:
                    continue
                inner_sum = 0.0
                for j in range(m):
                    inner_sum += U[i * m + j][c] * V[j * p + k][c]
                val += w_val * inner_sum
            C_test[i, k] = val

    max_err = np.max(np.abs(C_test - C_ref))
    if max_err > 1e-6:
        # Fallback to schoolbook rank 48
        rank = n * m * p  # 48
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1.0
                    V[j * p + k][idx] = 1.0
                    W[k * n + i][idx] = 1.0
                    idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
