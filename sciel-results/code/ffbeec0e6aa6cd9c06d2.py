import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 49 (Strassen 4x4x4) + 44 (4 * Rank(2x2x3) = 4 * 11)
    total_rank = 49 + 44
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    sub_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # p0
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # p1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # p2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # p3
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # p4
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # p5
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # p6
    ]
    
    for m_idx in range(7):
        x_def, y_def = strassen_terms[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = sub_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), c_x in x_sub.items():
                for b, coeff_block in x_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_idx = i * m + j
                    u_coeffs[u_idx] += c_x * coeff_block
            
            for (li, lj), c_y in y_sub.items():
                for b, coeff_block in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_idx = i * p + j
                    v_coeffs[v_idx] += c_y * coeff_block
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            
            cm_coeffs = {
                (0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0
            }
            
            if t_idx == 0:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 1:
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif t_idx == 2:
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 3:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif t_idx == 4:
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif t_idx == 5:
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 6:
                cm_coeffs[(0,0)] += 1.0
            
            c_block_coeffs = {}
            if m_idx == 0:
                c_block_coeffs = {(0,0): 1.0, (1,1): 1.0}
            elif m_idx == 1:
                c_block_coeffs = {(1,0): 1.0, (1,1): -1.0}
            elif m_idx == 2:
                c_block_coeffs = {(0,1): 1.0, (1,1): 1.0}
            elif m_idx == 3:
                c_block_coeffs = {(0,0): 1.0, (1,0): 1.0}
            elif m_idx == 4:
                c_block_coeffs = {(0,0): -1.0, (0,1): 1.0}
            elif m_idx == 5:
                c_block_coeffs = {(1,1): 1.0}
            elif m_idx == 6:
                c_block_coeffs = {(0,0): 1.0}
            
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                for (br, bc), cb_c in c_block_coeffs.items():
                    if cb_c == 0.0: continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c * cb_c
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]

    # --- Part 2: 4x4x3 using 4 * (2x2x3) (Rank 44) ---
    
    def add_2x2x3(a_row_block, a_col_block, b_row_block, c_row_block):
        nonlocal current_rank
        
        strassen_2x2x2_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}),            # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}),           # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}),           # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}),            # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}), # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})  # M7
        ]
        
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = strassen_2x2x2_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), c_x in x_sub.items():
                i = 2 * a_row_block + li
                j = 2 * a_col_block + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c_x
            
            for (li, lj), c_y in y_sub.items():
                i = 2 * b_row_block + li
                j = 4 + lj
                v_idx = i * p + j
                v_coeffs[v_idx] += c_y
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            
            cm_coeffs = {
                (0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0
            }
            
            if t_idx == 0:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 1:
                cm_coeffs[(1,0)] += 1.0
                cm_coeffs[(1,1)] += -1.0
            elif t_idx == 2:
                cm_coeffs[(0,1)] += 1.0
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 3:
                cm_coeffs[(0,0)] += 1.0
                cm_coeffs[(1,0)] += 1.0
            elif t_idx == 4:
                cm_coeffs[(0,0)] += -1.0
                cm_coeffs[(0,1)] += 1.0
            elif t_idx == 5:
                cm_coeffs[(1,1)] += 1.0
            elif t_idx == 6:
                cm_coeffs[(0,0)] += 1.0
            
            for (li, lj), cm_c in cm_coeffs.items():
                if cm_c == 0.0: continue
                i_final = 2 * c_row_block + li
                k_final = 4 + lj
                w_idx = k_final * n + i_final
                w_coeffs[w_idx] += cm_c
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]
        
        schoolbook_terms = [
            ({(0,0): 1}, {(0,0): 1}, (0,0)), # A00 B02 -> C00
            ({(0,1): 1}, {(1,0): 1}, (0,0)), # A01 B12 -> C00
            ({(1,0): 1}, {(0,0): 1}, (1,0)), # A10 B02 -> C10
            ({(1,1): 1}, {(1,0): 1}, (1,0))  # A11 B12 -> C10
        ]
        
        for t_idx in range(4):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_pos = schoolbook_terms[t_idx]
            
            u_coeffs = [0.0] * (n*m)
            v_coeffs = [0.0] * (m*p)
            
            for (li, lj), c_x in x_sub.items():
                i = 2 * a_row_block + li
                j = 2 * a_col_block + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c_x
            
            for (li, lj), c_y in y_sub.items():
                i = 2 * b_row_block + li
                j = 6
                v_idx = i * p + j
                v_coeffs[v_idx] += c_y
            
            for i in range(n*m):
                U[i][r] = u_coeffs[i]
            for j in range(m*p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p*n)
            i_final = 2 * c_row_block + c_pos[0]
            k_final = 6
            w_idx = k_final * n + i_final
            w_coeffs[w_idx] = 1.0
            
            for i in range(p*n):
                W[i][r] = w_coeffs[i]

    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)

    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
