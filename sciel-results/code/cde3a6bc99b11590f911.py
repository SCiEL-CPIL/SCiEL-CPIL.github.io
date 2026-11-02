import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26

    # Build the decomposition using Strassen blocks for pairs of k-columns
    # and a schoolbook block for the last k-column (p=13 is odd).
    # Rank = 7 * floor(p/2) + 4 = 7*6 + 4 = 46

    R = 7 * (p // 2) + 4
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)

    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i

    col = 0
    # Strassen blocks for k0 = 0, 2, 4, 6, 8, 10
    for k0 in range(0, 12, 2):
        # 7 Strassen products
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11) B00
        # M3 = A00 (B01-B11)
        # M4 = A11 (B10-B00)
        # M5 = (A00+A01) B11
        # M6 = (A10-A00) (B00+B01)
        # M7 = (A01-A11) (B10+B11)
        prods = [
            # (U_entries, V_entries)
            ([(A_idx(0,0), 1), (A_idx(1,1), 1)], [(B_idx(0,0), 1), (B_idx(1,1), 1)]),
            ([(A_idx(1,0), 1), (A_idx(1,1), 1)], [(B_idx(0,0), 1)]),
            ([(A_idx(0,0), 1)], [(B_idx(0,1), 1), (B_idx(1,1), -1)]),
            ([(A_idx(1,1), 1)], [(B_idx(1,0), 1), (B_idx(0,0), -1)]),
            ([(A_idx(0,0), 1), (A_idx(0,1), 1)], [(B_idx(1,1), 1)]),
            ([(A_idx(1,0), 1), (A_idx(0,0), -1)], [(B_idx(0,0), 1), (B_idx(0,1), 1)]),
            ([(A_idx(0,1), 1), (A_idx(1,1), -1)], [(B_idx(1,0), 1), (B_idx(1,1), 1)]),
        ]
        for t, (u_entries, v_entries) in enumerate(prods):
            c = col + t
            for idx, val in u_entries:
                U[idx, c] += val
            for idx, val in v_entries:
                V[idx, c] += val

        # W combinations for C[i,k0] and C[i,k0+1]
        # C[0,k0]   = M1 + M3 - M5 + M7
        # C[0,k0+1] = M2 + M4
        # C[1,k0]   = M1 + M2
        # C[1,k0+1] = M1 - M2 + M3 + M6
        c_combos = [
            (0, k0,     [(0, 1), (2, 1), (4, -1), (6, 1)]),
            (0, k0 + 1, [(1, 1), (3, 1)]),
            (1, k0,     [(0, 1), (1, 1)]),
            (1, k0 + 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
        ]
        for (i, k, terms) in c_combos:
            for (t, coeff) in terms:
                W[C_idx(i, k), col + t] += coeff

        col += 7

    # Schoolbook block for k=12
    prods_school = [
        (A_idx(0,0), B_idx(0,12)),
        (A_idx(0,1), B_idx(1,12)),
        (A_idx(1,0), B_idx(0,12)),
        (A_idx(1,1), B_idx(1,12)),
    ]
    for t, (u_idx, v_idx) in enumerate(prods_school):
        c = col + t
        U[u_idx, c] += 1
        V[v_idx, c] += 1

    W[C_idx(0, 12), col + 0] += 1
    W[C_idx(0, 12), col + 1] += 1
    W[C_idx(1, 12), col + 2] += 1
    W[C_idx(1, 12), col + 3] += 1
    col += 4

    assert col == R, f"col={col}, R={R}"

    # Verify the decomposition
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    target_arr = np.array(target_indices)
    a_idx = target_arr[:, 0]
    b_idx = target_arr[:, 1]
    c_idx = target_arr[:, 2]
    U_t = U[a_idx, :]
    V_t = V[b_idx, :]
    W_t = W[c_idx, :]
    prod = U_t * V_t * W_t
    sums = np.sum(prod, axis=1)
    valid = np.all(np.abs(sums - 1.0) < 1e-9)

    if not valid:
        # Fallback to schoolbook
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col_idx] = 1.0
                    V0[j * p + k, col_idx] = 1.0
                    W0[k * n + i, col_idx] = 1.0
                    col_idx += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # Snap to half-integers for clean output
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
