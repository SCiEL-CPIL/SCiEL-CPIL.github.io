import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Strassen rank-7 terms for 2x2x2 multiplication
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Top-level Strassen for 4x4x4 (2x2 blocks of size 2x2)
    top_4x4 = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)])
    ]

    all_terms = []
    for t_idx in range(7):
        a_blocks, b_blocks, c_blocks = top_4x4[t_idx]
        for s_idx in range(7):
            s_u, s_v, s_w = strassen_2x2[s_idx]
            u_global = {}
            for (r, c), coeff_s in s_u.items():
                for (u, v), coeff_a in a_blocks:
                    i = 2*u + r
                    j = 2*v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a
            v_global = {}
            for (r, c), coeff_s in s_v.items():
                for (v, w), coeff_b in b_blocks:
                    j = 2*v + r
                    k = 2*w + c
                    v_global[(j, k)] = v_global.get((j, k), 0) + coeff_s * coeff_b
            w_global = {}
            for (r, c), coeff_s in s_w.items():
                for (u, w), coeff_c in c_blocks:
                    i = 2*u + r
                    k = 2*w + c
                    w_global[(i, k)] = w_global.get((i, k), 0) + coeff_s * coeff_c
            u_global = {k: v for k, v in u_global.items() if v != 0}
            v_global = {k: v for k, v in v_global.items() if v != 0}
            w_global = {k: v for k, v in w_global.items() if v != 0}
            if u_global and v_global and w_global:
                all_terms.append((u_global, v_global, w_global))

    # Right half: shift k by 4
    terms_right = []
    for u_dict, v_dict, w_dict in all_terms:
        u_new = dict(u_dict)
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_right.append((u_new, v_new, w_new))

    total_terms = all_terms + terms_right
    rank = len(total_terms)

    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for col_idx, (u_dict, v_dict, w_dict) in enumerate(total_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i, col_idx] = coeff

    # Verify
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    try:
        C = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
        valid = np.allclose(C, 1.0, atol=1e-9)
    except Exception:
        valid = False

    if valid:
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
