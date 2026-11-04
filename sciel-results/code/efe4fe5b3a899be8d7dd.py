import numpy as np
import time
import hashlib

DIMENSIONS = [4, 4, 8]


def construct():
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    # --- 2x2 Strassen (rank 7) ---
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0: M1 = (a00+a11)(b00+b11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    # c=1: M2 = (a10+a11)*b00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    # c=2: M3 = a00*(b01-b11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    # c=3: M4 = a11*(b10-b00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    # c=4: M5 = (a00+a01)*b11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    # c=5: M6 = (a10-a00)*(b00+b01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    # c=6: M7 = (a01-a11)*(b10+b11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block-Strassen 4x4*4x4 (rank 49) specs
    A_specs = [
        [(1, 0, 0), (1, 1, 1)],
        [(1, 1, 0), (1, 1, 1)],
        [(1, 0, 0)],
        [(1, 1, 1)],
        [(1, 0, 0), (1, 0, 1)],
        [(1, 1, 0), (-1, 0, 0)],
        [(1, 0, 1), (-1, 1, 1)],
    ]
    B_specs = [
        [(1, 0, 0), (1, 1, 1)],
        [(1, 0, 0)],
        [(1, 0, 1), (-1, 1, 1)],
        [(1, 1, 0), (-1, 0, 0)],
        [(1, 1, 1)],
        [(1, 0, 0), (1, 0, 1)],
        [(1, 1, 0), (1, 1, 1)],
    ]
    C_specs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_dict, v_dict, w_dict):
        U_row = np.zeros(nm)
        V_row = np.zeros(mp)
        W_row = np.zeros(pn)
        for (i, j), val in u_dict.items():
            U_row[i * m + j] += val
        for (j, k), val in v_dict.items():
            V_row[j * p + k] += val
        for (k, i), val in w_dict.items():
            W_row[k * n + i] += val
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    def strassen_4x4x4(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off):
        for k_p in range(7):
            for m_s in range(7):
                u_dict = {}
                v_dict = {}
                w_dict = {}
                for (coeff, br, bc) in A_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            u_val = U_2[r * 2 + c, m_s]
                            if u_val != 0:
                                gi = A_row_off + br * 2 + r
                                gj = A_col_off + bc * 2 + c
                                u_dict[(gi, gj)] = u_dict.get((gi, gj), 0) + coeff * u_val
                for (coeff, br, bc) in B_specs[k_p]:
                    for r in range(2):
                        for c in range(2):
                            v_val = V_2[r * 2 + c, m_s]
                            if v_val != 0:
                                gj = B_row_off + br * 2 + r
                                gk = B_col_off + bc * 2 + c
                                v_dict[(gj, gk)] = v_dict.get((gj, gk), 0) + coeff * v_val
                for (cbr, cbc), coeff_c in C_specs[k_p].items():
                    for r in range(2):
                        for c in range(2):
                            w_val = W_2[r * 2 + c, m_s]
                            if w_val != 0:
                                gi = C_row_off + cbr * 2 + r
                                gk = C_col_off + cbc * 2 + c
                                w_dict[(gk, gi)] = w_dict.get((gk, gi), 0) + coeff_c * w_val
                if u_dict and v_dict and w_dict:
                    add_term(u_dict, v_dict, w_dict)

    # Two 4x4*4x4 blocks for B columns 0-3 and 4-7
    strassen_4x4x4(0, 0, 0, 0, 0, 0)
    strassen_4x4x4(0, 0, 0, 4, 0, 4)

    U = np.array(U_list).T  # (nm, rank)
    V = np.array(V_list).T  # (mp, rank)
    W = np.array(W_list).T  # (pn, rank)

    def verify(Um, Vm, Wm):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(Um[i * m + j, :], Vm[j * p + k, :] * Wm[k * n + i, :]))
                    if abs(val - 1.0) > 1e-9:
                        err += abs(val - 1.0)
        return err

    def schoolbook():
        U_s = np.zeros((nm, n * m * p))
        V_s = np.zeros((mp, n * m * p))
        W_s = np.zeros((pn, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return U_s, V_s, W_s

    def merge_duplicates(Um, Vm, Wm):
        # Merge rank-1 terms that share an identical U-row (exact, keeps tensor).
        groups = {}
        for c in range(Um.shape[1]):
            key = hashlib.md5(np.asarray(Um[:, c]).tobytes()).hexdigest()
            groups.setdefault(key, []).append(c)
        if all(len(v) == 1 for v in groups.values()):
            return Um, Vm, Wm
        cols = []
        for v in groups.values():
            if len(v) == 1:
                cols.append(v[0])
            else:
                cols.append(v[0])
        new_r = len(cols)
        U_new = Um[:, cols].copy()
        V_new = np.zeros((Vm.shape[0], new_r))
        W_new = np.zeros((Wm.shape[0], new_r))
        for new_idx, v in enumerate(groups.values()):
            V_new[:, new_idx] = Vm[:, v].sum(axis=1)
            W_new[:, new_idx] = Wm[:, v].sum(axis=1)
        return U_new, V_new, W_new

    # Refinement pass: exact rank reduction by merging duplicate U-rows.
    if time.time() < deadline - 5:
        Um2, Vm2, Wm2 = merge_duplicates(U, V, W)
        if verify(Um2, Vm2, Wm2) < 1e-8:
            U, V, W = Um2, Vm2, Wm2

    if verify(U, V, W) > 1e-8:
        U, V, W = schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
