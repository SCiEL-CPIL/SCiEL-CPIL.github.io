import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Strassen terms for 2x2x2
    # Each term: (U_dict, V_dict, W_dict) where keys are (row, col)
    strassen_2x2 = [
        # M1: (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Build 4x4x4 Strassen decomposition (rank 49)
    # Treat 4x4 as 2x2 blocks of 2x2 matrices
    # Top-level Strassen for 4x4x4
    top_4x4 = [
        # M1: (A00+A11)(B00+B11)
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2: (A10+A11)B00
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        # M3: A00(B01-B11)
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        # M4: A11(B10-B00)
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        # M5: (A00+A01)B11
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        # M6: (A10-A00)(B00+B01)
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M7: (A01-A11)(B10+B11)
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)])
    ]

    # Build all 49 terms for 4x4x4
    terms_4x4 = []
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
                terms_4x4.append((u_global, v_global, w_global))

    # Now extend to 4x4x8
    # For k=0..3: use terms_4x4 as-is
    # For k=4..7: shift k by 4 in V and W, keep U the same
    terms_full = []
    for u_dict, v_dict, w_dict in terms_4x4:
        terms_full.append((u_dict, v_dict, w_dict))
    
    for u_dict, v_dict, w_dict in terms_4x4:
        u_new = dict(u_dict)
        v_new = {}
        for (j, k), coeff in v_dict.items():
            v_new[(j, k + 4)] = coeff
        w_new = {}
        for (i, k), coeff in w_dict.items():
            w_new[(i, k + 4)] = coeff
        terms_full.append((u_new, v_new, w_new))

    rank = len(terms_full)
    
    # Build arrays
    U_arr = np.zeros((n * m, rank))
    V_arr = np.zeros((m * p, rank))
    W_arr = np.zeros((p * n, rank))
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(terms_full):
        for (i, j), coeff in u_dict.items():
            U_arr[i * m + j, col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            V_arr[j * p + k, col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            W_arr[k * n + i, col_idx] = coeff
    
    # Verify using batched computation
    valid = True
    if time.time() < deadline - 10:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(t_val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
    
    if not valid:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    print(f"Rank: {U.shape[1]}")
