import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] = coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] = coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def strassen_2x2(oi, oj, ok):
        u1 = {(oi, oj): 1, (oi+1, oj+1): 1}
        v1 = {(oj, ok): 1, (oj+1, ok+1): 1}
        w1 = {(oi, ok): 1, (oi+1, ok+1): 1}
        add_term(u1, v1, w1)
        
        u2 = {(oi+1, oj): 1, (oi+1, oj+1): 1}
        v2 = {(oj, ok): 1}
        w2 = {(oi+1, ok): 1, (oi+1, ok+1): -1}
        add_term(u2, v2, w2)
        
        u3 = {(oi, oj): 1}
        v3 = {(oj, ok+1): 1, (oj+1, ok+1): -1}
        w3 = {(oi, ok+1): 1, (oi+1, ok+1): 1}
        add_term(u3, v3, w3)
        
        u4 = {(oi+1, oj+1): 1}
        v4 = {(oj+1, ok): 1, (oj, ok): -1}
        w4 = {(oi, ok): 1, (oi+1, ok): 1}
        add_term(u4, v4, w4)
        
        u5 = {(oi, oj): 1, (oi, oj+1): 1}
        v5 = {(oj+1, ok+1): 1}
        w5 = {(oi, ok): -1, (oi, ok+1): 1}
        add_term(u5, v5, w5)
        
        u6 = {(oi+1, oj): 1, (oi, oj): -1}
        v6 = {(oj, ok): 1, (oj, ok+1): 1}
        w6 = {(oi+1, ok+1): 1}
        add_term(u6, v6, w6)
        
        u7 = {(oi, oj+1): 1, (oi+1, oj+1): -1}
        v7 = {(oj+1, ok): 1, (oj+1, ok+1): 1}
        w7 = {(oi, ok): 1}
        add_term(u7, v7, w7)
    
    def add_super_term(u_blocks, v_blocks, w_blocks, oi, oj, ok):
        terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]
        
        for u_t, v_t, w_t in terms:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for (br, bc), coeff_u in u_blocks.items():
                for (r, c), coeff_t in u_t.items():
                    i = oi + 2*br + r
                    j = oj + 2*bc + c
                    u_col[i * m + j] += coeff_u * coeff_t
            
            for (br, bc), coeff_v in v_blocks.items():
                for (r, c), coeff_t in v_t.items():
                    j = oj + 2*br + r
                    k = ok + 2*bc + c
                    v_col[j * p + k] += coeff_v * coeff_t
            
            for (br, bc), coeff_w in w_blocks.items():
                for (r, c), coeff_t in w_t.items():
                    i = oi + 2*br + r
                    k = ok + 2*bc + c
                    w_col[k * n + i] += coeff_w * coeff_t
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    def strassen_4x4(oi, oj, ok):
        add_super_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, oi, oj, ok)
        add_super_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}, oi, oj, ok)
        add_super_term({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, oi, oj, ok)
        add_super_term({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, oi, oj, ok)
        add_super_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}, oi, oj, ok)
        add_super_term({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}, oi, oj, ok)
        add_super_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}, oi, oj, ok)
    
    if _time_remaining() > 5.0:
        strassen_4x4(0, 0, 0)
    
    for rb in range(2):
        if _time_remaining() < 5.0:
            break
        strassen_2x2(rb*2, 0, 4)
        strassen_2x2(rb*2, 2, 4)
    
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
