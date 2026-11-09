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
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict { (i, j): coeff } for A
    # V_coeffs: dict { (j, k): coeff } for B
    # W_coeffs: dict { (i, k): coeff } for C
    strassen_terms_2x2 = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Top-level Strassen definitions for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    top_level_terms = [
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
        a_blocks, b_blocks, c_blocks = top_level_terms[t_idx]
        
        for s_idx in range(7):
            s_u, s_v, s_w = strassen_terms_2x2[s_idx]
            
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

    # Construct right half terms (shift k by 4)
    terms_right = []
    for u_dict, v_dict, w_dict in all_terms:
        u_new = u_dict
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_right.append((u_new, v_new, w_new))

    total_terms = all_terms + terms_right
    rank = len(total_terms)
    
    # Verify time
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
            # W is indexed by k, i in the tensor T_ijk = sum U_ij V_jk W_ki
            # The dict key is (i, k) from the construction where w_global[(i, k)] was set.
            # Wait, in the construction:
            # w_global[(i, k)] = ...
            # So the dict has keys (i, k).
            # The matrix W has shape (p*n, rank).
            # The index in W matrix is k*n + i.
            W[k * n + i, col_idx] = coeff

    # Verify 4x4x8 decomposition
    U_reshaped = U.reshape(4, 4, rank)
    V_reshaped = V.reshape(4, 8, rank)
    W_reshaped = W.reshape(8, 4, rank)
    
    try:
        C448 = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
        # The target tensor is all 1s
        valid = np.allclose(C448, 1.0, atol=1e-9)
    except Exception:
        valid = False

    if valid:
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
