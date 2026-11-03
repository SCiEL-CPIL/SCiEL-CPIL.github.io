import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n, m, p = DIMENSIONS
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for i, c in u_coeffs:
            if c != 0:
                U[i][col] += c
        for j, c in v_coeffs:
            if c != 0:
                V[j][col] += c
        for k, c in w_coeffs:
            if c != 0:
                W[k][col] += c
        col += 1

    def a_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)
        
    def b1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*6 + (y*2+c)
        
    def c1_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (y*2+c)*4 + (x*2+r)

    def get_strassen_2x2_terms():
        s1 = ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)])
        s2 = ([((1,0), 1), ((1,1), 1)], [((0,0), 1)])
        s3 = ([((0,0), 1)], [((0,1), 1), ((1,1), -1)])
        s4 = ([((1,1), 1)], [((1,0), 1), ((0,0), -1)])
        s5 = ([((0,0), 1), ((0,1), 1)], [((1,1), 1)])
        s6 = ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)])
        s7 = ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
        return [s1, s2, s3, s4, s5, s6, s7]

    strassen_2x2_terms = get_strassen_2x2_terms()
    
    result_map = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    big_combos = [
        ([('00', 1), ('11', 1)], [('00', 1), ('11', 1)]),
        ([('10', 1), ('11', 1)], [('00', 1)]),
        ([('00', 1)], [('01', 1), ('11', -1)]),
        ([('11', 1)], [('10', 1), ('00', -1)]),
        ([('00', 1), ('01', 1)], [('11', 1)]),
        ([('10', 1), ('00', -1)], [('00', 1), ('01', 1)]),
        ([('01', 1), ('11', -1)], [('10', 1), ('11', 1)])
    ]
    
    big_result_map = {
        '00': [(0, 1), (3, 1), (4, -1), (6, 1)],
        '01': [(2, 1), (4, 1)],
        '10': [(1, 1), (3, 1)],
        '11': [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b1_idx(b_block, ry, cy)
                    v_coeffs.append((idx, b_sign * cy_c))
            
            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                big_coeff = 0
                for m_idx, m_c in big_contribs:
                    if m_idx == big_idx:
                        big_coeff = m_c
                        break
                if big_coeff == 0:
                    continue
                
                for rc, s_contribs in result_map.items():
                    small_coeff = 0
                    for s_check, s_c in s_contribs:
                        if s_check == s_idx:
                            small_coeff = s_c
                            break
                    if small_coeff == 0:
                        continue
                    
                    idx = c1_idx(c_block, rc[0], rc[1])
                    w_coeffs.append((idx, big_coeff * small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
            
    products = [
        ('00', 'T', 'top'),
        ('01', 'B', 'top'),
        ('10', 'T', 'bot'),
        ('11', 'B', 'bot')
    ]
    
    for a_block, b_block, c_part in products:
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                x, y = int(a_block[0]), int(a_block[1])
                idx = (x*2+rx)*4 + (y*2+cx)
                u_coeffs.append((idx, cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                if b_block == 'T':
                    idx = ry*6 + (4+cy)
                else:
                    idx = (2+ry)*6 + (4+cy)
                v_coeffs.append((idx, cy_c))
            
            w_coeffs = []
            for rc, s_contribs in result_map.items():
                small_coeff = 0
                for s_check, s_c in s_contribs:
                    if s_check == s_idx:
                        small_coeff = s_c
                        break
                if small_coeff == 0:
                    continue
                
                if c_part == 'top':
                    idx = (4+rc[1])*4 + rc[0]
                else:
                    idx = (4+rc[1])*4 + (2+rc[0])
                
                w_coeffs.append((idx, small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    # Verify decomposition correctness
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Tensor equation: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Compute decomposed tensor
    T_dec = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        # u_col[i*m+j], v_col[j*p+k], w_col[k*n+i]
        # Build outer product
        for i in range(n):
            for j in range(m):
                u_val = u_col[i*m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j*p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k*n + i]
                    T_dec[i, j, k] += u_val * v_val * w_val
    
    T_true = np.ones((n, m, p))
    err = np.max(np.abs(T_dec - T_true))
    
    if err > 1e-10:
        # Decomposition invalid; return as-is (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    # Try to prune: remove columns that are exactly zero
    keep = []
    for r in range(rank):
        if np.any(U_np[:, r] != 0) and np.any(V_np[:, r] != 0) and np.any(W_np[:, r] != 0):
            keep.append(r)
    
    if len(keep) < rank:
        U = [[U[i][r] for r in keep] for i in range(n*m)]
        V = [[V[i][r] for r in keep] for i in range(m*p)]
        W = [[W[i][r] for r in keep] for i in range(p*n)]
    
    # Try to find and remove linearly dependent columns
    # If column r can be expressed as a linear combination of other columns in all three matrices
    # simultaneously, it can be removed. This is a strong condition.
    # Simpler: try removing each column and check if the tensor is still satisfied
    # by solving for the removed column's contribution being zero.
    # This is expensive, so only do it if we have time.
    if time.time() < deadline - 10:
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        cur_rank = U_np.shape[1]
        
        # Try greedy removal
        removed = set()
        for r in range(cur_rank):
            if r in removed:
                continue
            # Skip column r, check if remaining columns still satisfy tensor
            # Build tensor from remaining columns
            T_test = np.zeros((n, m, p))
            for rr in range(cur_rank):
                if rr == r or rr in removed:
                    continue
                u_col = U_np[:, rr]
                v_col = V_np[:, rr]
                w_col = W_np[:, rr]
                for i in range(n):
                    for j in range(m):
                        u_val = u_col[i*m + j]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            v_val = v_col[j*p + k]
                            if v_val == 0:
                                continue
                            w_val = w_col[k*n + i]
                            T_test[i, j, k] += u_val * v_val * w_val
            
            if np.max(np.abs(T_test - T_true)) < 1e-10:
                removed.add(r)
        
        if removed:
            keep2 = [r for r in range(cur_rank) if r not in removed]
            U = [[U[i][r] for r in keep2] for i in range(n*m)]
            V = [[V[i][r] for r in keep2] for i in range(m*p)]
            W = [[W[i][r] for r in keep2] for i in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
