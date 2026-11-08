"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
rank-2 decompositions for 2x1x2 and 1x2x2 blocks, targeting rank 48.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    # Strategy: Decompose C = A*B where A is 3x3, B is 3x8.
    # For each pair of columns k0 in {0,2,4,6}:
    #   A[0:2,0:2] * B[0:2,k0:k0+2]: Strassen 2x2x2 (rank 7)
    #   A[0:2,2]   * B[2,k0:k0+2]:   2x1x2 (rank 2)
    #   A[2,0:2]   * B[0:2,k0:k0+2]: 1x2x2 (rank 2)
    #   A[2,2]     * B[2,k0:k0+2]:   1x1x2 (rank 2)
    # Per column-pair: 7+2+2+2 = 13, total 4*13 = 52
    # Wait, 1x1x2 is rank 2 (two independent products).
    # Actually let me reconsider: can we do better?
    #
    # Alternative: combine the 2x1x2 and 1x2x2 and 1x1x2 parts.
    # For a fixed column-pair, the non-Strassen part involves:
    #   A[0:2,2]*B[2,k0], A[0:2,2]*B[2,k0+1] -> 4 terms (rank 2 for 2x1x2)
    #   A[2,0:2]*B[0:2,k0], A[2,0:2]*B[0:2,k0+1] -> 4 terms (rank 2 for 1x2x2)
    #   A[2,2]*B[2,k0], A[2,2]*B[2,k0+1] -> 2 terms (rank 2 for 1x1x2)
    # Total non-Strassen per pair: 6, grand total: 4*7 + 4*6 = 28+24 = 52

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
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

    def add_2x1x2(row_off_a, col_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Decompose A[rows, col] * B[row, cols] -> C[rows, cols] (2x1x2, rank 2).
        A is 2x1 at (row_off_a:row_off_a+2, col_a), B is 1x2 at (row_off_b, col_off_b:col_off_b+2).
        C[i,k] = A[row_off_a+i, col_a] * B[row_off_b, col_off_b+k]
        This is just two independent products, rank 2.
        """
        # Term 1: A[row_off_a, col_a] * B[row_off_b, col_off_b] -> C[row_off_a, col_off_b]
        U_t.append([((row_off_a, col_a), 1)])
        V_t.append([((row_off_b, col_off_b), 1)])
        W_t.append([((row_off_a, col_off_b), 1)])

        # Term 2: A[row_off_a+1, col_a] * B[row_off_b, col_off_b+1] -> C[row_off_a+1, col_off_b+1]
        U_t.append([((row_off_a + 1, col_a), 1)])
        V_t.append([((row_off_b, col_off_b + 1), 1)])
        W_t.append([((row_off_a + 1, col_off_b + 1), 1)])

    def add_1x2x2(row_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Decompose A[row, cols] * B[rows, cols] -> C[row, cols] (1x2x2, rank 2).
        A is 1x2 at (row_a, col_off_a:col_off_a+2), B is 2x2 at (row_off_b:row_off_b+2, col_off_b:col_off_b+2).
        C[row, col_off_b+k] = sum_j A[row, col_off_a+j] * B[row_off_b+j, col_off_b+k]
        This is two independent dot products of length 2, rank 2.
        """
        # For k=0: C[row, col_off_b] = A[row, col_off_a]*B[row_off_b, col_off_b] + A[row, col_off_a+1]*B[row_off_b+1, col_off_b]
        # For k=1: C[row, col_off_b+1] = A[row, col_off_a]*B[row_off_b, col_off_b+1] + A[row, col_off_a+1]*B[row_off_b+1, col_off_b+1]
        # We can factor: (A[row,col_off_a]*B[row_off_b,:] + A[row,col_off_a+1]*B[row_off_b+1,:])
        # This is rank 2: one term for each output column.
        # Term for k=0:
        U_t.append([((row_a, col_off_a), 1), ((row_a, col_off_a + 1), 1)])
        V_t.append([((row_off_b, col_off_b), 1), ((row_off_b + 1, col_off_b), 1)])
        W_t.append([((row_a, col_off_b), 1)])

        # Term for k=1:
        U_t.append([((row_a, col_off_a), 1), ((row_a, col_off_a + 1), 1)])
        V_t.append([((row_off_b, col_off_b + 1), 1), ((row_off_b + 1, col_off_b + 1), 1)])
        W_t.append([((row_a, col_off_b + 1), 1)])

    def add_1x1x2(row_a, col_a, row_b, col_off_b, U_t, V_t, W_t):
        """Decompose A[row,col] * B[row, cols] -> C[row, cols] (1x1x2, rank 2).
        C[row, col_off_b+k] = A[row_a, col_a] * B[row_b, col_off_b+k]
        """
        # Term for k=0:
        U_t.append([((row_a, col_a), 1)])
        V_t.append([((row_b, col_off_b), 1)])
        W_t.append([((row_a, col_off_b), 1)])

        # Term for k=1:
        U_t.append([((row_a, col_a), 1)])
        V_t.append([((row_b, col_off_b + 1), 1)])
        W_t.append([((row_a, col_off_b + 1), 1)])

    U_t = []
    V_t = []
    W_t = []

    for k0 in [0, 2, 4, 6]:
        # Strassen: A[0:2,0:2] * B[0:2, k0:k0+2]
        strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)

        # 2x1x2: A[0:2, 2] * B[2, k0:k0+2]
        add_2x1x2(0, 2, 2, k0, U_t, V_t, W_t)

        # 1x2x2: A[2, 0:2] * B[0:2, k0:k0+2]
        add_1x2x2(2, 0, 0, k0, U_t, V_t, W_t)

        # 1x1x2: A[2,2] * B[2, k0:k0+2]
        add_1x1x2(2, 2, 2, k0, U_t, V_t, W_t)

    rank = len(W_t)

    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Convert to nice integer/half-integer values
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
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

    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
