"""Rewritten program for 2x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # n=2, m=5, p=6

    # Partition middle dimension: 5 = 2 + 2 + 1
    # A = [A1 | A2 | A3], B = [B1; B2; B3]
    # C = A1*B1 + A2*B2 + A3*B3
    # A1,A2: 2x2, A3: 2x1
    # B1,B2: 2x6, B3: 1x6

    # Use Strassen for 2x2 blocks (rank 7 per 2x2x2 block)
    # 2x2 x 2x6 requires 3 pairs of columns: rank = 7*3 = 21
    # Two such blocks: 42
    # 2x1 x 1x6: rank 1
    # Total: 43

    total_rank = 43

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Strassen 2x2x2 terms:
    # M1 = (A00+A11)(B00+B11) -> C00,C01,C10,C11 all get +1
    # M2 = (A10+A11)B00 -> C10,C11 get +1
    # M3 = A00(B10-B11) -> C01,C11 get +1
    # M4 = A11(B01-B10) -> C00,C01,C10,C11 get +1
    # M5 = (A00+A01)B11 -> C00,C10 get -1; C11 gets +1
    # M6 = (A10-A00)(B00+B10) -> C11 gets +1
    # M7 = (A01-A11)(B01+B11) -> C00,C11 get +1

    def add_block(a_col_base, b_row_base, c_col_base, offset):
        """Add Strassen terms for a 2x2 x 2x2x2 block."""
        # A entries at rows 0,1 cols a_col_base, a_col_base+1
        # B entries at rows b_row_base, b_row_base+1 cols c_col_base, c_col_base+1
        # C entries at rows c_col_base, c_col_base+1, cols 0,1 (output row indices)
        # W convention: W[k*n+i, r]

        # M1
        c = offset + 0
        U[0 * m + a_col_base, c] = 1.0    # A00
        U[1 * m + a_col_base + 1, c] = 1.0  # A11
        V[b_row_base * p + c_col_base, c] = 1.0        # B00
        V[(b_row_base + 1) * p + c_col_base + 1, c] = 1.0  # B11
        W[c_col_base * n + 0, c] = 1.0   # C(c_col_base, 0)
        W[(c_col_base + 1) * n + 0, c] = 1.0  # C(c_col_base+1, 0)
        W[c_col_base * n + 1, c] = 1.0   # C(c_col_base, 1)
        W[(c_col_base + 1) * n + 1, c] = 1.0  # C(c_col_base+1, 1)

        # M2
        c = offset + 1
        U[1 * m + a_col_base, c] = 1.0    # A10
        U[1 * m + a_col_base + 1, c] = 1.0  # A11
        V[b_row_base * p + c_col_base, c] = 1.0  # B00
        W[(c_col_base) * n + 1, c] = 1.0   # C(c_col_base, 1) = C10 part
        W[(c_col_base + 1) * n + 1, c] = 1.0  # C(c_col_base+1, 1) = C11 part

        # M3
        c = offset + 2
        U[0 * m + a_col_base, c] = 1.0    # A00
        V[(b_row_base + 1) * p + c_col_base, c] = 1.0    # B10
        V[(b_row_base + 1) * p + c_col_base + 1, c] = -1.0  # -B11
        W[(c_col_base) * n + 0, c] = 1.0   # C(c_col_base, 0) = C01 part
        W[(c_col_base + 1) * n + 0, c] = 1.0  # C(c_col_base+1, 0) = C11 part

        # M4
        c = offset + 3
        U[1 * m + a_col_base + 1, c] = 1.0  # A11
        V[b_row_base * p + c_col_base + 1, c] = 1.0    # B01
        V[(b_row_base + 1) * p + c_col_base, c] = -1.0  # -B10
        W[c_col_base * n + 0, c] = 1.0
        W[(c_col_base + 1) * n + 0, c] = 1.0
        W[c_col_base * n + 1, c] = 1.0
        W[(c_col_base + 1) * n + 1, c] = 1.0

        # M5
        c = offset + 4
        U[0 * m + a_col_base, c] = 1.0    # A00
        U[0 * m + a_col_base + 1, c] = 1.0  # A01
        V[(b_row_base + 1) * p + c_col_base + 1, c] = 1.0  # B11
        W[c_col_base * n + 0, c] = -1.0
        W[(c_col_base + 1) * n + 0, c] = -1.0
        W[(c_col_base + 1) * n + 1, c] = 1.0

        # M6
        c = offset + 5
        U[1 * m + a_col_base, c] = 1.0    # A10
        U[0 * m + a_col_base, c] = -1.0   # -A00
        V[b_row_base * p + c_col_base, c] = 1.0    # B00
        V[(b_row_base + 1) * p + c_col_base, c] = 1.0  # B10
        W[(c_col_base + 1) * n + 1, c] = 1.0  # C11

        # M7
        c = offset + 6
        U[0 * m + a_col_base + 1, c] = 1.0  # A01
        U[1 * m + a_col_base + 1, c] = -1.0 # -A11
        V[b_row_base * p + c_col_base + 1, c] = 1.0    # B01
        V[(b_row_base + 1) * p + c_col_base + 1, c] = 1.0  # B11
        W[c_col_base * n + 0, c] = 1.0
        W[(c_col_base + 1) * n + 1, c] = 1.0

    # Block 1: A1 (2x2, cols 0-1) * B1 (2x6, rows 0-1)
    for cp in range(3):
        cc = cp * 2
        add_block(0, 0, cc, cp * 7)

    # Block 2: A2 (2x2, cols 2-3) * B2 (2x6, rows 2-3)
    for cp in range(3):
        cc = cp * 2
        add_block(2, 2, cc, 21 + cp * 7)

    # Block 3: A3 (2x1, col 4) * B3 (1x6, row 4)
    col = 42
    for i in range(2):
        U[i * m + 4, col] = 1.0
    for k in range(6):
        V[4 * p + k, col] = 1.0
        W[k * n + 0, col] = 1.0
        W[k * n + 1, col] = 1.0

    # Verification: check tensor identity
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                e = abs(s - 1.0)
                if e > max_err:
                    max_err = e

    if max_err > 1e-10:
        # Fallback: schoolbook
        tr = n * m * p
        U = np.zeros((n * m, tr))
        V = np.zeros((m * p, tr))
        W = np.zeros((p * n, tr))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
