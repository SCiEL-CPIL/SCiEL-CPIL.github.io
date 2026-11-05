import time

_START = time.time()
_DEADLINE = 115.0  # safety margin before 120s kill

def construct():
    n, m, p = 4, 5, 5  # A is n×m, B is m×p, C is n×p
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Part 1: 4x4x4 multiplication (i∈{0..3}, k∈{0..3}, j∈{0..3}) using 2-level nested Strassen
    # Inner Strassen for 2x2x2: 7 products
    inner_strassen = [
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,1)]},
        {'A': [(1,0,1),(1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1),(1,1,-1)]},
        {'A': [(0,0,1)], 'B': [(0,1,1),(1,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        {'A': [(1,1,1)], 'B': [(1,0,1),(0,0,-1)], 'C': [(0,0,1),(1,0,1)]},
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1),(0,1,1)]},
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(1,1,1)]},
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(0,0,1)]},
    ]
    
    # Outer Strassen for 4x4x4
    outer_strassen = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1, 3: 1}, 'C': {0: 1, 3: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}, 'C': {2: 1, 3: -1}},
        {'A': {0: 1}, 'B': {1: 1, 3: -1}, 'C': {1: 1, 3: 1}},
        {'A': {3: 1}, 'B': {2: 1, 0: -1}, 'C': {0: 1, 2: 1}},
        {'A': {0: 1, 1: 1}, 'B': {3: 1}, 'C': {0: -1, 1: 1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}, 'C': {3: 1}},
        {'A': {1: 1, 3: -1}, 'B': {2: 1, 3: 1}, 'C': {0: 1}},
    ]
    
    def a_global(block, li, lk):
        if block == 0: return li, lk
        elif block == 1: return li, lk + 2
        elif block == 2: return li + 2, lk
        elif block == 3: return li + 2, lk + 2
    
    def b_global(block, lk, lj):
        if block == 0: return lk, lj
        elif block == 1: return lk, lj + 2
        elif block == 2: return lk + 2, lj
        elif block == 3: return lk + 2, lj + 2
    
    def c_global(block, li, lj):
        if block == 0: return li, lj
        elif block == 1: return li, lj + 2
        elif block == 2: return li + 2, lj
        elif block == 3: return li + 2, lj + 2
    
    # Generate 49 terms for the 4x4x4 part
    for prod_idx, prod in enumerate(outer_strassen):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = prod['C']
        
        for inner_idx, inner in enumerate(inner_strassen):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for a_block, a_sign in a_form.items():
                for li, lk, inner_sign in inner['A']:
                    gi, gk = a_global(a_block, li, lk)
                    u_vec[gi * m + gk] += a_sign * inner_sign
            
            for b_block, b_sign in b_form.items():
                for lk, lj, inner_sign in inner['B']:
                    gk, gj = b_global(b_block, lk, lj)
                    v_vec[gk * p + gj] += b_sign * inner_sign
            
            for c_block, c_sign in c_contrib.items():
                for li, lj, inner_sign in inner['C']:
                    gi, gj = c_global(c_block, li, lj)
                    w_vec[gj * n + gi] += c_sign * inner_sign
            
            add_rank1(u_vec, v_vec, w_vec)
        
        if time.time() - _START > _DEADLINE:
            return {"U": U_cols, "V": V_cols, "W": W_cols}
    
    if time.time() - _START > _DEADLINE:
        return {"U": U_cols, "V": V_cols, "W": W_cols}
    
    # Part 2: 4x4x1 (i∈{0..3}, k∈{0..3}, j=4) using Strassen
    # 7 Strassen products, each is 2x2x1 = 2 schoolbook terms, total 14 terms
    strassen_4x4x1 = [
        {'A': {0: 1, 3: 1}, 'B': {0: 1}, 'C': {0: 1}},
        {'A': {2: 1, 3: 1}, 'B': {0: 1}, 'C': {1: 1}},
        {'A': {0: 1}, 'B': {1: 1, 0: -1}, 'C': {}},
        {'A': {3: 1}, 'B': {1: 1, 0: -1}, 'C': {0: 1, 1: 1}},
        {'A': {0: 1, 1: 1}, 'B': {1: 1}, 'C': {0: -1}},
        {'A': {2: 1, 0: -1}, 'B': {0: 1, 1: 1}, 'C': {}},
        {'A': {1: 1, 3: -1}, 'B': {0: 1, 1: 1}, 'C': {0: 1}},
    ]
    
    for prod_idx, prod in enumerate(strassen_4x4x1):
        a_form = prod['A']
        b_form = prod['B']
        c_contrib = prod['C']
        
        for li in range(2):
            for lk in range(2):
                u_vec = [0.0] * (n * m)
                v_vec = [0.0] * (m * p)
                w_vec = [0.0] * (p * n)
                
                a_coeff = 0
                for a_block, a_sign in a_form.items():
                    if a_block == 0 and li < 2 and lk < 2: a_coeff += a_sign
                    elif a_block == 1 and li < 2 and lk >= 2: a_coeff += a_sign
                    elif a_block == 2 and li >= 2 and lk < 2: a_coeff += a_sign
                    elif a_block == 3 and li >= 2 and lk >= 2: a_coeff += a_sign
                
                b_coeff = 0
                for b_block, b_sign in b_form.items():
                    if b_block == 0 and lk < 2: b_coeff += b_sign
                    elif b_block == 1 and lk >= 2: b_coeff += b_sign
                
                c_coeff = 0
                for c_block, c_sign in c_contrib.items():
                    if c_block == 0 and li < 2: c_coeff += c_sign
                    elif c_block == 1 and li >= 2: c_coeff += c_sign
                
                if a_coeff == 0 or b_coeff == 0 or c_coeff == 0:
                    continue
                
                gi, gk, gj = li, lk, 4
                u_vec[gi * m + gk] = a_coeff
                v_vec[gk * p + gj] = b_coeff
                w_vec[gj * n + gi] = c_coeff
                
                add_rank1(u_vec, v_vec, w_vec)
        
        if time.time() - _START > _DEADLINE:
            return {"U": U_cols, "V": V_cols, "W": W_cols}
    
    if time.time() - _START > _DEADLINE:
        return {"U": U_cols, "V": V_cols, "W": W_cols}
    
    # Part 3: 4x1x5 (i∈{0..3}, k=4, j∈{0..4}) schoolbook = 20 terms
    for i in range(n):
        for j in range(p):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[i * m + 4] = 1.0
            v_vec[4 * p + j] = 1.0
            w_vec[j * n + i] = 1.0
            
            add_rank1(u_vec, v_vec, w_vec)
    
    return {"U": U_cols, "V": V_cols, "W": W_cols}
