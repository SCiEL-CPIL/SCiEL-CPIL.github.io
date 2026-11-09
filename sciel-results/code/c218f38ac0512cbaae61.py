import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

_DECOMP = None
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120

def _build_decomposition():
    n, m, p = 4, 4, 7
    
    rank_strassen = 49
    rank_rect = 24  # rank for 4x4x3
    total_rank = rank_strassen + rank_rect
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 73
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 73
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 73
    
    col = 0
    # Strassen 49 for 4x4x4
    u_in = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    v_in = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]
    ]
    u_out = u_in[:]
    v_out = v_in[:]
    w_outer = w_out[:]
    
    for k_outer in range(7):
        for l_inner in range(7):
            u_vec = [0] * 16
            for i_inner in range(2):
                for j_inner in range(2):
                    ci = u_in[l_inner][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = u_out[k_outer][b]
                        if co == 0:
                            continue
                        br, bc = b // 2, b % 2
                        ig = 2 * br + i_inner
                        jg = 2 * bc + j_inner
                        u_vec[ig * 4 + jg] += ci * co
            
            v_vec = [0] * 28
            for i_inner in range(2):
                for j_inner in range(2):
                    ci = v_in[l_inner][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = v_out[k_outer][b]
                        if co == 0:
                            continue
                        br, bc = b // 2, b % 2
                        jg = 2 * br + i_inner
                        kg = 2 * bc + j_inner
                        v_vec[jg * 7 + kg] += ci * co
            
            w_vec = [0] * 28
            for cb in range(4):
                cow = w_outer[cb][k_outer]
                if cow == 0:
                    continue
                for ci2 in range(4):
                    ciw = w_out[ci2][l_inner]
                    if ciw == 0:
                        continue
                    tc = cow * ciw
                    if tc == 0:
                        continue
                    cbr, cbc = cb // 2, cb % 2
                    cir, cic = ci2 // 2, ci2 % 2
                    ig = 2 * cbr + cir
                    kg = 2 * cbc + cic
                    w_vec[kg * 4 + ig] += tc
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            col += 1
    
    # Rank-24 decomposition for 4x4x3
    # 4 sub-problems, each 2x2 x 2x3 with rank 6
    for sub in range(4):
        # Determine offsets
        if sub == 0:  # A11 * B1 -> C[0:2, 0:3]
            a_row_off = 0
            a_col_off = 0
            b_row_off = 0
            c_row_off = 0
        elif sub == 1:  # A12 * B2 -> C[0:2, 0:3]
            a_row_off = 0
            a_col_off = 2
            b_row_off = 2
            c_row_off = 0
        elif sub == 2:  # A21 * B1 -> C[2:4, 0:3]
            a_row_off = 2
            a_col_off = 0
            b_row_off = 0
            c_row_off = 2
        else:  # sub == 3: A22 * B2 -> C[2:4, 0:3]
            a_row_off = 2
            a_col_off = 2
            b_row_off = 2
            c_row_off = 2
        
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[(a_row_off + i) * 4 + (a_col_off + j)][col] = 1
                    V[(b_row_off + j) * 3 + k][col] = 1
                    W[k * 4 + (c_row_off + i)][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

_DECOMP = _build_decomposition()

def construct():
    if time.time() > _DEADLINE:
        return _DECOMP
    return _DECOMP
