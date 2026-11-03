import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 tensor: 7 rank-1 terms
    strassen_2x2 = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # S1
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,1)]),         # S2
        ([(0,1)], [(1,1),(3,-1)], [(0,-1),(1,1)]),       # S3
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,-1)]),       # S4
        ([(0,1),(1,1)], [(3,1)], [(1,1),(3,1)]),         # S5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,-1),(2,1)]), # S6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,-1),(3,1)]), # S7
    ]
    
    c_coeffs = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},
        (0,1): {2:1, 4:1},
        (1,0): {1:1, 3:1},
        (1,1): {0:1, 1:-1, 2:1, 5:1},
    }
    
    total_rank = 77
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= total_rank:
            return
        for idx, c in u_coeffs:
            if c != 0 and 0 <= idx < n*m:
                U[idx][col] += c
        for idx, c in v_coeffs:
            if c != 0 and 0 <= idx < m*p:
                V[idx][col] += c
        for idx, c in w_coeffs:
            if c != 0 and 0 <= idx < p*n:
                W[idx][col] += c
        col += 1
    
    big_combos = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
    ]
    
    big_result_map = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},
        (0,1): {2:1, 4:1},
        (1,0): {1:1, 3:1},
        (1,1): {0:1, 1:-1, 2:1, 5:1},
    }
    
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        if time.time() > deadline:
            break
        for s_idx in range(7):
            if time.time() > deadline:
                break
            u_small, v_small, w_small = strassen_2x2[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in u_small:
                for (bx, by, sign) in a_parts:
                    idx = (bx*2+rx)*4 + (by*2+cx)
                    u_coeffs.append((idx, sign * cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in v_small:
                for (bx, by, sign) in b_parts:
                    idx = (bx*2+ry)*6 + (by*2+cy)
                    v_coeffs.append((idx, sign * cy_c))
            
            w_coeffs = []
            for (cx, cy), contribs in big_result_map.items():
                big_coeff = contribs.get(big_idx, 0)
                if big_coeff == 0:
                    continue
                for (rx, cx2), s_contribs in c_coeffs.items():
                    small_coeff = s_contribs.get(s_idx, 0)
                    if small_coeff == 0:
                        continue
                    idx = (cy*2+cx2)*4 + (cx*2+rx)
                    w_coeffs.append((idx, big_coeff * small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    products = [
        ((0,0), 'top'),
        ((0,1), 'top'),
        ((1,0), 'bot'),
        ((1,1), 'bot'),
    ]
    
    for (ax, ay), part in products:
        if time.time() > deadline:
            break
        for s_idx in range(7):
            if time.time() > deadline:
                break
            u_small, v_small, w_small = strassen_2x2[s_idx]
            
            u_coeffs = []
            for (rx, cx), cx_c in u_small:
                idx = (ax*2+rx)*4 + (ay*2+cx)
                u_coeffs.append((idx, cx_c))
            
            v_coeffs = []
            for (ry, cy), cy_c in v_small:
                if part == 'top':
                    idx = ry*6 + (4+cy)
                else:
                    idx = (2+ry)*6 + (4+cy)
                v_coeffs.append((idx, cy_c))
            
            w_coeffs = []
            for (rx, cx2), s_contribs in c_coeffs.items():
                small_coeff = s_contribs.get(s_idx, 0)
                if small_coeff == 0:
                    continue
                if part == 'top':
                    idx = (4+cx2)*4 + rx
                else:
                    idx = (4+cx2)*4 + (2+rx)
                w_coeffs.append((idx, small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
    
    # Try to find linear dependencies to reduce rank
    if time.time() < deadline - 10 and col < total_rank:
        try:
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Each term t has vectors u_t, v_t, w_t
            # A linear combination sum_i alpha_i * (u_i, v_i, w_i) = 0
            # means for all (i,j,k): sum_t alpha_t * U[i,t]*V[j,t]*W[k,t] = 0
            # This is nonlinear in alpha. But we can look for cases where
            # the u, v, or w vectors are linearly dependent.
            
            # Check if any subset of terms has linearly dependent U vectors
            # (or V or W vectors). If so, we might be able to eliminate terms.
            
            # For now, just return what we have
            pass
        except:
            pass
    
    return {"U": U, "V": V, "W": W}
