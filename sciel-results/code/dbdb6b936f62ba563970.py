"""
Strategy: Direct Strassen-based tensor decomposition for 2x4 * 4x8.
We split the middle dimension m=4 into two blocks of 2, and the output dimension p=8
into four blocks of 2. This gives 8 independent 2x2 * 2x2 multiplications, each
handled by the rank-7 Strassen decomposition. Total rank = 8 * 7 = 56.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-56 decomposition for 2x4 * 4x8 using Strassen's algorithm.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Strassen decomposition for 2x2 * 2x2 (rank 7)
    # Local indices: A[r,c], B[r,c], C[r,c] with r,c in {0,1}
    # A local idx: r*2+c; B local idx: r*2+c; C local idx: r*2+c
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    strassen_terms = [
        # M1 = (a00+a11)*(b00+b11) -> C00 += 1, C11 += 1
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        # M2 = (a10+a11)*b00 -> C10 += 1, C11 += -1
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        # M3 = a00*(b01-b11) -> C01 += 1, C11 += 1
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        # M4 = a11*(b10-b00) -> C00 += 1, C10 += 1
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        # M5 = (a00+a01)*b11 -> C00 += -1, C01 += 1
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        # M6 = (a10-a00)*(b00+b01) -> C11 += 1
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        # M7 = (a01-a11)*(b10+b11) -> C00 += 1
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]}
    ]

    rank = 56
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Global index helpers
    # A is 2x4: A[i, j] -> idx i*4 + j
    def a_gidx(i, j):
        return i * 4 + j

    # B is 4x8: B[i, j] -> idx i*8 + j
    def b_gidx(i, j):
        return i * 8 + j

    # C is 2x8: C[i, k] -> W idx k*2 + i (trace convention W[k*n+i])
    def w_gidx(i, k):
        return k * 2 + i

    col = 0
    # 8 sub-problems:
    # A1 (cols 0-1) * B1 (rows 0-1) * C1 (cols 0-1)
    # A1 (cols 0-1) * B1 (rows 0-1) * C2 (cols 2-3)
    # A1 (cols 0-1) * B1 (rows 0-1) * C3 (cols 4-5)
    # A1 (cols 0-1) * B1 (rows 0-1) * C4 (cols 6-7)
    # A2 (cols 2-3) * B2 (rows 2-3) * C1 (cols 0-1)
    # A2 (cols 2-3) * B2 (rows 2-3) * C2 (cols 2-3)
    # A2 (cols 2-3) * B2 (rows 2-3) * C3 (cols 4-5)
    # A2 (cols 2-3) * B2 (rows 2-3) * C4 (cols 6-7)

    # For each sub-problem:
    # A block: 2x2, col offset a_col_off (0 or 2)
    # B block: 2x2, row offset b_row_off (0 or 2), col offset b_col_off (0,2,4,6)
    # C block: 2x2, col offset c_col_off (same as b_col_off)

    sub_problems = [
        # (a_col_off, b_row_off, b_col_off, c_col_off)
        (0, 0, 0, 0),  # A1*B1[0:2,0:2] -> C[0:2,0:2]
        (0, 0, 2, 2),  # A1*B1[0:2,2:4] -> C[0:2,2:4]
        (0, 0, 4, 4),  # A1*B1[0:2,4:6] -> C[0:2,4:6]
        (0, 0, 6, 6),  # A1*B1[0:2,6:8] -> C[0:2,6:8]
        (2, 2, 0, 0),  # A2*B2[2:4,0:2] -> C[0:2,0:2]
        (2, 2, 2, 2),  # A2*B2[2:4,2:4] -> C[0:2,2:4]
        (2, 2, 4, 4),  # A2*B2[2:4,4:6] -> C[0:2,4:6]
        (2, 2, 6, 6),  # A2*B2[2:4,6:8] -> C[0:2,6:8]
    ]

    for (a_col_off, b_row_off, b_col_off, c_col_off) in sub_problems:
        for term in strassen_terms:
            # U part: A indices
            for (local_idx, coeff) in term['u']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gj = a_col_off + lc
                g_idx = a_gidx(gi, gj)
                U[g_idx, col] += coeff

            # V part: B indices
            for (local_idx, coeff) in term['v']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = b_row_off + lr
                gj = b_col_off + lc
                g_idx = b_gidx(gi, gj)
                V[g_idx, col] += coeff

            # W part: C indices
            for (local_idx, coeff) in term['w']:
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gk = c_col_off + lc
                g_idx = w_gidx(gi, gk)
                W[g_idx, col] += coeff

            col += 1

    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p, num_tests=10):
        rng = np.random.default_rng(42)
        for _ in range(num_tests):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            expected = A @ B
            computed = np.zeros((n, p))
            for c in range(U_mat.shape[1]):
                u_vec = U_mat[:, c]
                v_vec = V_mat[:, c]
                w_vec = W_mat[:, c]
                u_reshaped = u_vec.reshape(n, m)
                v_reshaped = v_vec.reshape(m, p)
                w_reshaped = w_vec.reshape(p, n)
                uv = u_reshaped @ v_reshaped
                w_transposed = w_reshaped.T
                computed += uv * w_transposed
            if not np.allclose(computed, expected, atol=1e-10, rtol=1e-10):
                return False
        return True

    def to_half_integer_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2
                row_list.append(rounded)
            result.append(row_list)
        return result

    if verify(U, V, W, n, m, p):
        return {
            "U": to_half_integer_list(U),
            "V": to_half_integer_list(V),
            "W": to_half_integer_list(W)
        }
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": to_half_integer_list(U),
            "V": to_half_integer_list(V),
            "W": to_half_integer_list(W)
        }
# EVOLVE-BLOCK-END
