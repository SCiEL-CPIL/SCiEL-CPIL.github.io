import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        for idx, coeff in u_vals.items():
            u_col[idx] = coeff
        for idx, coeff in v_vals.items():
            v_col[idx] = coeff
        for idx, coeff in w_vals.items():
            w_col[idx] = coeff
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Part 1: Schoolbook for C_right (k in 4..5)
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                u_vals = {i * 4 + j: 1.0}
                v_vals = {j * 6 + k: 1.0}
                w_vals = {k * 4 + i: 1.0}
                add_term(u_vals, v_vals, w_vals)

    # Part 2: Strassen for C_left (k in 0..3)
    # A blocks:
    # A11: i 0-1, j 0-1
    # A12: i 0-1, j 2-3
    # A21: i 2-3, j 0-1
    # A22: i 2-3, j 2-3
    
    # B_left blocks:
    # B11: j 0-1, k 0-1
    # B12: j 0-1, k 2-3
    # B21: j 2-3, k 0-1
    # B22: j 2-3, k 2-3
    
    # C_left blocks:
    # C11: i 0-1, k 0-1
    # C12: i 0-1, k 2-3
    # C21: i 2-3, k 0-1
    # C22: i 2-3, k 2-3
    
    # Helper to get A index
    def a_idx(i, j): return i * 4 + j
    # Helper to get B index
    def b_idx(j, k): return j * 6 + k
    # Helper to get C index
    def c_idx(i, k): return k * 4 + i
    
    # Define Strassen X and Y blocks as lists of (block_name, sub_i, sub_j, coeff)
    # But we need linear forms.
    
    # X1 = A11 + A22
    # X2 = A21 + A22
    # X3 = A11
    # X4 = A22
    # X5 = A11 + A12
    # X6 = A21 - A11
    # X7 = A12 - A22
    
    # Y1 = B11 + B22
    # Y2 = B11
    # Y3 = B12 - B22
    # Y4 = B21 - B11
    # Y5 = B22
    # Y6 = B11 + B12
    # Y7 = B21 + B22
    
    # We will iterate through the 7 products.
    # For each product M_l = X_l Y_l, we expand into 8 scalar products.
    # M_l[p, q] = sum_u X_l[p, u] Y_l[u, q]
    
    # We need to represent X_l and Y_l as 2x2 matrices of linear forms.
    # A linear form is a dict {a_idx: coeff} or {b_idx: coeff}.
    
    def get_A_form(block, pi, pj, coeff):
        # block is 11, 12, 21, 22
        # pi, pj are 0 or 1
        d = {}
        if block == 11:
            i_base, j_base = 0, 0
        elif block == 12:
            i_base, j_base = 0, 2
        elif block == 21:
            i_base, j_base = 2, 0
        else: # 22
            i_base, j_base = 2, 2
            
        i = i_base + pi
        j = j_base + pj
        d[a_idx(i, j)] = coeff
        return d

    def get_B_form(block, pi, pj, coeff):
        # block is 11, 12, 21, 22
        # pi is row of B (j), pj is col of B (k)
        d = {}
        if block == 11:
            j_base, k_base = 0, 0
        elif block == 12:
            j_base, k_base = 0, 2
        elif block == 21:
            j_base, k_base = 2, 0
        else: # 22
            j_base, k_base = 2, 2
            
        j = j_base + pi
        k = k_base + pj
        d[b_idx(j, k)] = coeff
        return d

    def add_forms(f1, f2, coeff1=1.0, coeff2=1.0):
        res = {}
        for k, v in f1.items():
            res[k] = res.get(k, 0.0) + v * coeff1
        for k, v in f2.items():
            res[k] = res.get(k, 0.0) + v * coeff2
        return res

    # Define X_l as 2x2 list of dicts
    # X1
    X1 = [
        [add_forms(get_A_form(11,0,0,1), get_A_form(22,0,0,1)), add_forms(get_A_form(11,0,1,1), get_A_form(22,0,1,1))],
        [add_forms(get_A_form(11,1,0,1), get_A_form(22,1,0,1)), add_forms(get_A_form(11,1,1,1), get_A_form(22,1,1,1))]
    ]
    # X2 = A21 + A22
    X2 = [
        [add_forms(get_A_form(21,0,0,1), get_A_form(22,0,0,1)), add_forms(get_A_form(21,0,1,1), get_A_form(22,0,1,1))],
        [add_forms(get_A_form(21,1,0,1), get_A_form(22,1,0,1)), add_forms(get_A_form(21,1,1,1), get_A_form(22,1,1,1))]
    ]
    # X3 = A11
    X3 = [
        [get_A_form(11,0,0,1), get_A_form(11,0,1,1)],
        [get_A_form(11,1,0,1), get_A_form(11,1,1,1)]
    ]
    # X4 = A22
    X4 = [
        [get_A_form(22,0,0,1), get_A_form(22,0,1,1)],
        [get_A_form(22,1,0,1), get_A_form(22,1,1,1)]
    ]
    # X5 = A11 + A12
    X5 = [
        [add_forms(get_A_form(11,0,0,1), get_A_form(12,0,0,1)), add_forms(get_A_form(11,0,1,1), get_A_form(12,0,1,1))],
        [add_forms(get_A_form(11,1,0,1), get_A_form(12,1,0,1)), add_forms(get_A_form(11,1,1,1), get_A_form(12,1,1,1))]
    ]
    # X6 = A21 - A11
    X6 = [
        [add_forms(get_A_form(21,0,0,1), get_A_form(11,0,0,-1)), add_forms(get_A_form(21,0,1,1), get_A_form(11,0,1,-1))],
        [add_forms(get_A_form(21,1,0,1), get_A_form(11,1,0,-1)), add_forms(get_A_form(21,1,1,1), get_A_form(11,1,1,-1))]
    ]
    # X7 = A12 - A22
    X7 = [
        [add_forms(get_A_form(12,0,0,1), get_A_form(22,0,0,-1)), add_forms(get_A_form(12,0,1,1), get_A_form(22,0,1,-1))],
        [add_forms(get_A_form(12,1,0,1), get_A_form(22,1,0,-1)), add_forms(get_A_form(12,1,1,1), get_A_form(22,1,1,-1))]
    ]
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    
    # Y1 = B11 + B22
    Y1 = [
        [add_forms(get_B_form(11,0,0,1), get_B_form(22,0,0,1)), add_forms(get_B_form(11,0,1,1), get_B_form(22,0,1,1))],
        [add_forms(get_B_form(11,1,0,1), get_B_form(22,1,0,1)), add_forms(get_B_form(11,1,1,1), get_B_form(22,1,1,1))]
    ]
    # Y2 = B11
    Y2 = [
        [get_B_form(11,0,0,1), get_B_form(11,0,1,1)],
        [get_B_form(11,1,0,1), get_B_form(11,1,1,1)]
    ]
    # Y3 = B12 - B22
    Y3 = [
        [add_forms(get_B_form(12,0,0,1), get_B_form(22,0,0,-1)), add_forms(get_B_form(12,0,1,1), get_B_form(22,0,1,-1))],
        [add_forms(get_B_form(12,1,0,1), get_B_form(22,1,0,-1)), add_forms(get_B_form(12,1,1,1), get_B_form(22,1,1,-1))]
    ]
    # Y4 = B21 - B11
    Y4 = [
        [add_forms(get_B_form(21,0,0,1), get_B_form(11,0,0,-1)), add_forms(get_B_form(21,0,1,1), get_B_form(11,0,1,-1))],
        [add_forms(get_B_form(21,1,0,1), get_B_form(11,1,0,-1)), add_forms(get_B_form(21,1,1,1), get_B_form(11,1,1,-1))]
    ]
    # Y5 = B22
    Y5 = [
        [get_B_form(22,0,0,1), get_B_form(22,0,1,1)],
        [get_B_form(22,1,0,1), get_B_form(22,1,1,1)]
    ]
    # Y6 = B11 + B12
    Y6 = [
        [add_forms(get_B_form(11,0,0,1), get_B_form(12,0,0,1)), add_forms(get_B_form(11,0,1,1), get_B_form(12,0,1,1))],
        [add_forms(get_B_form(11,1,0,1), get_B_form(12,1,0,1)), add_forms(get_B_form(11,1,1,1), get_B_form(12,1,1,1))]
    ]
    # Y7 = B21 + B22
    Y7 = [
        [add_forms(get_B_form(21,0,0,1), get_B_form(22,0,0,1)), add_forms(get_B_form(21,0,1,1), get_B_form(22,0,1,1))],
        [add_forms(get_B_form(21,1,0,1), get_B_form(22,1,0,1)), add_forms(get_B_form(21,1,1,1), get_B_form(22,1,1,1))]
    ]
    
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # C reconstruction
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M_l[p, q] contributes to C blocks.
    # We need to add the term for each scalar product in M_l.
    
    for l in range(7):
        X_l = Xs[l]
        Y_l = Ys[l]
        
        for p in range(2):
            for q in range(2):
                # M_l[p, q] = sum_u X_l[p, u] Y_l[u, q]
                for u in range(2):
                    u_vals = X_l[p][u]
                    v_vals = Y_l[u][q]
                    
                    # Determine C contribution
                    w_vals = {}
                    
                    # C11
                    if l == 0 or l == 3 or l == 4 or l == 6: # M1, M4, M5, M7
                        coeff = 1.0
                        if l == 4: coeff = -1.0 # M5
                        # C11 is i 0-1, k 0-1
                        # p, q are 0-1
                        # i = p, k = q
                        c_i = p
                        c_k = q
                        w_vals[c_idx(c_i, c_k)] = w_vals.get(c_idx(c_i, c_k), 0.0) + coeff
                    # C12
                    if l == 2 or l == 4: # M3, M5
                        coeff = 1.0
                        # C12 is i 0-1, k 2-3
                        c_i = p
                        c_k = 2 + q
                        w_vals[c_idx(c_i, c_k)] = w_vals.get(c_idx(c_i, c_k), 0.0) + coeff
                    # C21
                    if l == 1 or l == 3: # M2, M4
                        coeff = 1.0
                        # C21 is i 2-3, k 0-1
                        c_i = 2 + p
                        c_k = q
                        w_vals[c_idx(c_i, c_k)] = w_vals.get(c_idx(c_i, c_k), 0.0) + coeff
                    # C22
                    if l == 0 or l == 1 or l == 2 or l == 5: # M1, M2, M3, M6
                        coeff = 1.0
                        if l == 1: coeff = -1.0 # M2
                        # C22 is i 2-3, k 2-3
                        c_i = 2 + p
                        c_k = 2 + q
                        w_vals[c_idx(c_i, c_k)] = w_vals.get(c_idx(c_i, c_k), 0.0) + coeff
                        
                    if w_vals:
                        add_term(u_vals, v_vals, w_vals)

    # Convert to rows
    U = [[U_cols[r][i] for r in range(len(U_cols))] for i in range(n * m)]
    V = [[V_cols[r][i] for r in range(len(V_cols))] for i in range(m * p)]
    W = [[W_cols[r][i] for r in range(len(W_cols))] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
