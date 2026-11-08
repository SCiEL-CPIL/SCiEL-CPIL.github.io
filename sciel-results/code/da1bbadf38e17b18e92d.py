"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses structured block decomposition: Strassen for 2x2x2 blocks
plus direct terms for the remaining rows/columns."""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

def construct():
    n, m, p = 3, 3, 8  # 3x3 * 3x8
    start_time = time.time()
    deadline = 140  # seconds, safety margin

    def check_time():
        return (time.time() - start_time) < deadline

    U_t = []
    V_t = []
    W_t = []

    def add_strassen(row_off_a, col_off_a, row_off_b, col_off_b):
        """Add Strassen 7-term decomposition for 2x2 * 2x2 block."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)

        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)

        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])

        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])

        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])

        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])

        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])

        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])

        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    def add_direct(i, j, k):
        """Add a direct term A[i,j]*B[j,k] -> C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])

    # Apply Strassen for each 2x2 block in the output
    # A[0:2, 0:2] * B[0:2, k:k+2] for k in {0, 2, 4, 6}
    for k0 in [0, 2, 4, 6]:
        if not check_time():
            break
        add_strassen(0, 0, 0, k0)

    # Direct terms for A[0:2, 2] * B[2, :]
    for i in range(2):
        for k in range(8):
            if not check_time():
                break
            add_direct(i, 2, k)

    # Direct terms for A[2, 0:2] * B[0:2, :]
    for j in range(2):
        for k in range(8):
            if not check_time():
                break
            add_direct(2, j, k)

    # Direct term for A[2, 2] * B[2, :]
    for k in range(8):
        if not check_time():
            break
        add_direct(2, 2, k)

    rank = len(W_t)

    # Build the factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i, idx] = coeff

    # Verify correctness
    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U, V, W):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Convert to nice integer/half-integer values
    def to_nice_list(arr):
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result

    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
