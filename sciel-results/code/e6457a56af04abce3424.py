import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 140

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen coefficients for 2x2x2
    # U[i, j, r], V[j, k, r], W[k, i, r]
    
    # r=1: M1 = (A00+A11)(B00+B11)
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    
    # r=2: M2 = (A10+A11)B00
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    
    # r=3: M3 = A00(B10-B11)
    u3 = {(0,0): 1}
    v3 = {(1,0): 1, (1,1): -1}
    
    # r=4: M4 = A11(B01-B00)
    u4 = {(1,1): 1}
    v4 = {(0,1): 1, (0,0): -1}
    
    # r=5: M5 = (A00+A01)B11
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    
    # r=6: M6 = (A10-A00)(B00+B01)
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    
    # r=7: M7 = (A01-A11)(B10+B11)
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    
    strassen_terms = [
        (u1, v1),
        (u2, v2),
        (u3, v3),
        (u4, v4),
        (u5, v5),
        (u6, v6),
        (u7, v7)
    ]

    # 4 blocks:
    # s=0: j_block=0, k_block=0
    # s=1: j_block=0, k_block=1
    # s=2: j_block=1, k_block=0
    # s=3: j_block=1, k_block=1
    
    for s in range(4):
        j_block = s // 2
        k_block = s % 2
        rank_offset = s * 7
        
        # Build local U (2x2x7) and V (2x2x7)
        U_loc = np.zeros((2, 2, 7))
        V_loc = np.zeros((2, 2, 7))
        
        for r_idx in range(7):
            u_dict, v_dict = strassen_terms[r_idx]
            for (i, j_local), val in u_dict.items():
                U_loc[i, j_local, r_idx] = val
            for (j_local, k_local), val in v_dict.items():
                V_loc[j_local, k_local, r_idx] = val
            
            # Fill global U and V
            for (i, j_local), val in u_dict.items():
                j_global = 2 * j_block + j_local
                u_idx = i * m + j_global
                r_global = rank_offset + r_idx
                U[u_idx, r_global] += val
                
            for (j_local, k_local), val in v_dict.items():
                j_global = 2 * j_block + j_local
                k_global = 2 * k_block + k_local
                v_idx = j_global * p + k_global
                r_global = rank_offset + r_idx
                V[v_idx, r_global] += val
        
        # Solve for W_loc (2x2x7)
        # T_loc[i, j, k] = 1 for all i, j, k
        # Equation: sum_r U_loc[i, j, r] * V_loc[j, k, r] * W_loc[k, i, r] = 1
        # This is linear in W.
        # Variables: W_loc[k, i, r] for k in 0..1, i in 0..1, r in 0..6. Total 28 vars.
        # Equations: for each i, j, k (8 equations).
        
        A_mat = np.zeros((8, 28))
        b_vec = np.ones(8)
        
        eq_idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Coefficients for W_loc[k, i, r]
                    for r in range(7):
                        var_idx = (k * 2 + i) * 7 + r # Indexing W as [k, i, r]
                        A_mat[eq_idx, var_idx] = U_loc[i, j, r] * V_loc[j, k, r]
                    b_vec[eq_idx] = 1
                    eq_idx += 1
        
        # Solve least squares
        W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        
        # Reshape W_flat to (2, 2, 7) and snap to half-integers
        W_loc = W_flat.reshape((2, 2, 7))
        
        # Snap to nearest 0.5
        W_loc_snapped = np.round(W_loc * 2) / 2
        
        # Verify and fill global W
        for k in range(2):
            for i in range(2):
                for r in range(7):
                    val = W_loc_snapped[k, i, r]
                    k_global = 2 * k_block + k
                    w_idx = k_global * n + i
                    r_global = rank_offset + r
                    W[w_idx, r_global] += val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
