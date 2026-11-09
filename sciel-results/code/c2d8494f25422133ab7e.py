import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(term_id, A_dict, B_dict, C_dict):
        for (i, j), coeff in A_dict.items():
            U[i * m + j][term_id] = coeff
        for (j, l), coeff in B_dict.items():
            V[j * p + l][term_id] = coeff
        for (i, l), coeff in C_dict.items():
            W[l * n + i][term_id] = coeff
            
    def get_A_block(i, j):
        return {(di, dj): {(i + di, j + dj): 1.0} for di in range(2) for dj in range(2)}
        
    def get_B_block(j, l, l_offset=0):
        return {(di, dj): {(j + di, l + dj + l_offset): 1.0} for di in range(2) for dj in range(2)}
        
    def combine(d1, d2, sign=1):
        res = {}
        for key, val in d1.items():
            res[key] = val.copy()
        for key, val in d2.items():
            if key in res:
                for k, v in val.items():
                    res[key][k] = res[key].get(k, 0.0) + sign * v
            else:
                res[key] = {k: sign * v for k, v in val.items()}
        return res
        
    A11 = get_A_block(0, 0)
    A12 = get_A_block(0, 1)
    A21 = get_A_block(1, 0)
    A22 = get_A_block(1, 1)
    
    M1_A = combine(A11, A22)
    M2_A = combine(A21, A22)
    M3_A = A11
    M4_A = A22
    M5_A = combine(A11, A12)
    M6_A = combine(A21, A11, -1)
    M7_A = combine(A12, A22, -1)
    
    C_coeff = {
        0: [1, 0, 0, 1],
        1: [0, 0, 1, -1],
        2: [0, 1, 0, 1],
        3: [1, 0, 1, 0],
        4: [-1, 1, 0, 0],
        5: [0, 0, 0, 1],
        6: [1, 0, 0, 0]
    }
    
    S_coeff = {
        0: [1, 0, 0, 0],
        1: [0, 0, 1, -1],
        2: [0, 1, 0, 1],
        3: [1, 0, 1, 0],
        4: [-1, 1, 0, 0],
        5: [0, 0, 0, 1],
        6: [1, 0, 0, 0]
    }
    
    def generate(l_offset, term_offset):
        B11 = get_B_block(0, 0, l_offset)
        B12 = get_B_block(0, 1, l_offset)
        B21 = get_B_block(1, 0, l_offset)
        B22 = get_B_block(1, 1, l_offset)
        
        M1_B = combine(B11, B22)
        M2_B = B11
        M3_B = combine(B12, B22, -1)
        M4_B = combine(B21, B11, -1)
        M5_B = B22
        M6_B = combine(B11, B12)
        M7_B = combine(B21, B22)
        
        M_forms = [
            (M1_A, M1_B),
            (M2_A, M2_B),
            (M3_A, M3_B),
            (M4_A, M4_B),
            (M5_A, M5_B),
            (M6_A, M6_B),
            (M7_A, M7_B)
        ]
        
        for s in range(7):
            L_A, L_B = M_forms[s]
            
            P_A = [
                combine(L_A[(0,0)], L_A[(1,1)]),
                combine(L_A[(1,0)], L_A[(1,1)]),
                L_A[(0,0)],
                L_A[(1,1)],
                combine(L_A[(0,0)], L_A[(0,1)]),
                combine(L_A[(1,0)], L_A[(0,0)], -1),
                combine(L_A[(0,1)], L_A[(1,1)], -1)
            ]
            P_B = [
                combine(L_B[(0,0)], L_B[(1,1)]),
                L_B[(0,0)],
                combine(L_B[(0,1)], L_B[(1,1)], -1),
                combine(L_B[(1,0)], L_B[(0,0)], -1),
                L_B[(1,1)],
                combine(L_B[(0,0)], L_B[(0,1)]),
                combine(L_B[(1,0)], L_B[(1,1)])
            ]
            
            for t in range(7):
                term_id = term_offset + s * 7 + t
                
                A_dict = {}
                for key, val in P_A[t].items():
                    for k, v in val.items():
                        A_dict[k] = A_dict.get(k, 0.0) + v
                        
                B_dict = {}
                for key, val in P_B[t].items():
                    for k, v in val.items():
                        B_dict[k] = B_dict.get(k, 0.0) + v
                        
                C_dict = {}
                for block_idx in range(4):
                    coeff_block = C_coeff[s][block_idx]
                    if coeff_block == 0: continue
                    i_base, l_base = (0,0) if block_idx==0 else (0,2) if block_idx==1 else (2,0) if block_idx==2 else (2,2)
                    l_base += l_offset
                    
                    for di in range(2):
                        for dj in range(2):
                            coeff_sub = S_coeff[t][(di, dj)]
                            if coeff_sub == 0: continue
                            total_coeff = coeff_block * coeff_sub
                            if total_coeff == 0: continue
                            i = i_base + di
                            l = l_base + dj
                            C_dict[(i, l)] = C_dict.get((i, l), 0.0) + total_coeff
                            
                add_term(term_id, A_dict, B_dict, C_dict)
                
    generate(0, 0)
    generate(4, 49)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
