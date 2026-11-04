"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition."""
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
    
    # Strassen 2x2x2 terms: (u_pattern, v_pattern, w_pattern)
    # Each pattern is a dict from (row, col) to coeff
    STRASSEN_2x2 = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    def strassen_2x2(oi, oj, ok):
        """Add Strassen for 2x2*2x2 at offset (oi,oj,ok)."""
        for u_pat, v_pat, w_pat in STRASSEN_2x2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            for (r, c), cf in u_pat.items():
                u_col[(oi+r) * m + (oj+c)] = cf
            for (r, c), cf in v_pat.items():
                v_col[(oj+r) * p + (ok+c)] = cf
            for (r, c), cf in w_pat.items():
                w_col[(ok+c) * n + (oi+r)] = cf
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    def add_super(ub, vb, wb, oi, oj, ok):
        """Add a super-term: sum of block products via Strassen."""
        for u_pat, v_pat, w_pat in STRASSEN_2x2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            for (br, bc), cf_u in ub.items():
                for (r, c), cf_t in u_pat.items():
                    u_col[(oi+2*br+r) * m + (oj+2*bc+c)] += cf_u * cf_t
            for (br, bc), cf_v in vb.items():
                for (r, c), cf_t in v_pat.items():
                    v_col[(oj+2*br+r) * p + (ok+2*bc+c)] += cf_v * cf_t
            for (br, bc), cf_w in wb.items():
                for (r, c), cf_t in w_pat.items():
                    w_col[(ok+2*bc+c) * n + (oi+2*br+r)] += cf_w * cf_t
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # 4x4x4 recursive Strassen (rank 49)
    strassen_4x4_blocks = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    for ub, vb, wb in strassen_4x4_blocks:
        if _time_remaining() < 5.0:
            break
        add_super(ub, vb, wb, 0, 0, 0)
    
    # 4x4x2: four 2x2x2 Strassen (rank 28)
    for rb in range(2):
        for cb in range(2):
            if _time_remaining() < 5.0:
                break
            strassen_2x2(rb*2, cb*2, 4)
    
    rank = len(U_cols)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
