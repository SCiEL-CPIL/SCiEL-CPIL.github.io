"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for remaining dimensions, targeting rank 68.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def strassen_block(ra, ca, rb, cb):
        """Return (U_list, V_list, W_list) for 2x2*2x2 Strassen block.
        U_list: list of (row_idx, coeff) for 2x2 A block
        V_list: list of (col_idx, coeff) for 2x2 B block  
        W_list: list of (out_idx, coeff) for 2x2 C block
        Indices are local to the 2x2 blocks (0..3).
        """
        # Local indices: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
        # e=(0,0), f=(0,1), g=(1,0), h=(1,1)
        # C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
        U_list = []
        V_list = []
        W_list = []

        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_list.append([(0, 1), (3, 1)])
        V_list.append([(0, 1), (3, 1)])
        W_list.append([(0, 1), (3, 1)])

        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_list.append([(2, 1), (3, 1)])
        V_list.append([(0, 1)])
        W_list.append([(2, 1), (3, -1)])

        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_list.append([(0, 1)])
        V_list.append([(1, 1), (3, -1)])
        W_list.append([(1, 1), (3, 1)])

        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_list.append([(3, 1)])
        V_list.append([(2, 1), (0, -1)])
        W_list.append([(0, 1), (2, 1)])

        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_list.append([(0, 1), (1, 1)])
        V_list.append([(3, 1)])
        W_list.append([(0, -1), (1, 1)])

        # m6 = (c-a)*(e+f) -> C11 += 1
        U_list.append([(2, 1), (0, -1)])
        V_list.append([(0, 1), (1, 1)])
        W_list.append([(3, 1)])

        # m7 = (b-d)*(g+h) -> C00 += 1
        U_list.append([(1, 1), (3, -1)])
        V_list.append([(2, 1), (3, 1)])
        W_list.append([(0, 1)])

        return U_list, V_list, W_list

    # Build terms: each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of ((i,j), coeff) for A[i,j]
    # V_coeffs: list of ((j,k), coeff) for B[j,k]
    # W_coeffs: list of ((i,k), coeff) for C[i,k]

    terms = []

    # Strassen blocks: A[0:2, 0:2] * B[0:2, k:k+2] for k in 0,2,4,6
    for k_off in [0, 2, 4, 6]:
        U_local, V_local, W_local = strassen_block(0, 0, 0, k_off)
        for u_loc, v_loc, w_loc in zip(U_local, V_local, W_local):
            U_full = []
            V_full = []
            W_full = []
            for (li, c) in u_loc:
                # li is local 2x2 index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                li_row = li // 2
                li_col = li % 2
                U_full.append(((li_row, li_col), c))
            for (lj, c) in v_loc:
                lj_row = lj // 2
                lj_col = lj % 2
                V_full.append(((lj_row, lj_col + k_off), c))
            for (lk, c) in w_loc:
                lk_row = lk // 2
                lk_col = lk % 2
                W_full.append(((lk_row, lk_col + k_off), c))
            terms.append((U_full, V_full, W_full))

    # Direct terms: A[0:2, 2] * B[2, k] for i in 0,1 and k in 0..7
    for i in range(2):
        for k in range(8):
            terms.append(([((i, 2), 1)], [((2, k), 1)], [((i, k), 1)]))

    # Direct terms: A[2, 0:2] * B[0:2, k] for j in 0,1 and k in 0..7
    for j in range(2):
        for k in range(8):
            terms.append(([((2, j), 1)], [((j, k), 1)], [((2, k), 1)]))

    # Direct terms: A[2, 2] * B[2, k] for k in 0..7
    for k in range(8):
        terms.append(([((2, 2), 1)], [((2, k), 1)], [((2, k), 1)]))

    rank = len(terms)

    # Build factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for idx in range(rank):
        u_t, v_t, w_t = terms[idx]
        for (i, j), coeff in u_t:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in v_t:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in w_t:
            W[k * n + i][idx] = coeff

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Build tensor from factors: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape for efficient computation
    T_computed = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]

    # Expected tensor: T[i,j,k] = 1 for all i,j,k
    if not np.allclose(T_computed, 1.0, atol=1e-9):
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
                    nice_row.append(float(val))
            result.append(nice_row)
        return result

    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
