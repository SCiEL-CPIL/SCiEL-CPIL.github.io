"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses a structured decomposition combining Strassen for 2x2x2 blocks
with direct terms, targeting rank 68.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def strassen_block(k_off):
        """Return 7 terms for 2x2 A-block times 2x2 B-block starting at column k_off.
        Each term is (U_list, V_list, W_list) with local indices.
        Local index mapping for 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        """
        terms_local = []
        
        # m1 = (a+d)(e+h)
        terms_local.append((
            [(0, 1), (3, 1)],
            [(0, 1), (3, 1)],
            [(0, 1), (3, 1)]
        ))
        # m2 = (c+d)*e
        terms_local.append((
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(2, 1), (3, -1)]
        ))
        # m3 = a*(f-h)
        terms_local.append((
            [(0, 1)],
            [(1, 1), (3, -1)],
            [(1, 1), (3, 1)]
        ))
        # m4 = d*(g-e)
        terms_local.append((
            [(3, 1)],
            [(2, 1), (0, -1)],
            [(0, 1), (2, 1)]
        ))
        # m5 = (a+b)*h
        terms_local.append((
            [(0, 1), (1, 1)],
            [(3, 1)],
            [(0, -1), (1, 1)]
        ))
        # m6 = (c-a)*(e+f)
        terms_local.append((
            [(2, 1), (0, -1)],
            [(0, 1), (1, 1)],
            [(3, 1)]
        ))
        # m7 = (b-d)*(g+h)
        terms_local.append((
            [(1, 1), (3, -1)],
            [(2, 1), (3, 1)],
            [(0, 1)]
        ))
        
        # Map local indices to global indices
        terms_global = []
        for u_loc, v_loc, w_loc in terms_local:
            U_full = []
            V_full = []
            W_full = []
            for (li, c) in u_loc:
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
            terms_global.append((U_full, V_full, W_full))
        return terms_global

    terms = []

    # Strassen blocks: A[0:2, 0:2] * B[0:2, k:k+2] for k in 0,2,4,6
    for k_off in [0, 2, 4, 6]:
        terms.extend(strassen_block(k_off))

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
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for idx in range(rank):
        u_t, v_t, w_t = terms[idx]
        for (i, j), coeff in u_t:
            U[i * m + j, idx] = coeff
        for (j, k), coeff in v_t:
            V[j * p + k, idx] = coeff
        for (i, k), coeff in w_t:
            W[k * n + i, idx] = coeff

    # Verify correctness using efficient tensor contraction
    T_computed = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # Reshape w to (n, p) for easier multiplication: w[k,i] -> w[i,k]
        w_col_t = w_col.T  # shape (n, p), w_col_t[i, k] = w[k, i]
        T_computed += u_col[:, :, np.newaxis] * v_col[np.newaxis, :, :] * w_col_t[:, np.newaxis, :]

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

    return {"U": to_nice_list(U), "V": to_nice_list(V), "W": to_nice_list(W)}
# EVOLVE-BLOCK-END
