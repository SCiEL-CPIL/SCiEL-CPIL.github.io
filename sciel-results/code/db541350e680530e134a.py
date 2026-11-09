def construct():
    n, m, p = 4, 4, 5
    rank = 49 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a linear form
    def add_form(form1, form2, sign=1):
        res = {}
        for k, v in form1.items():
            res[k] = res.get(k, 0) + v
        for k, v in form2.items():
            res[k] = res.get(k, 0) + sign * v
        # Remove zeros
        return {k: v for k, v in res.items() if v != 0}
        
    def mul_form(form1, form2):
        # This is not multiplication of forms, this is just to identify the pair
        pass

    # Define Strassen for 2x2
    # Inputs: X, Y are 2x2 lists of dicts (linear forms)
    # Outputs: 7 pairs of (L_A, L_B) and the mapping to result matrix entries
    
    def strassen_2x2(X, Y):
        # X, Y are 2x2 lists of dicts
        # s1 = (X00+X11)(Y00+Y11)
        # s2 = (X10+X11)Y00
        # s3 = X00(Y01-Y11)
        # s4 = X11(Y10-Y00)
        # s5 = (X00+X01)Y11
        # s6 = (X10-X00)(Y00+Y01)
        # s7 = (X01-X11)(Y10+Y11)
        
        terms = []
        
        # s1
        LA = add_form(X[0][0], X[1][1])
        LB = add_form(Y[0][0], Y[1][1])
        terms.append((LA, LB))
        
        # s2
        LA = add_form(X[1][0], X[1][1])
        LB = Y[0][0]
        terms.append((LA, LB))
        
        # s3
        LA = X[0][0]
        LB = add_form(Y[0][1], Y[1][1], -1)
        terms.append((LA, LB))
        
        # s4
        LA = X[1][1]
        LB = add_form(Y[1][0], Y[0][0], -1)
        terms.append((LA, LB))
        
        # s5
        LA = add_form(X[0][0], X[0][1])
        LB = Y[1][1]
        terms.append((LA, LB))
        
        # s6
        LA = add_form(X[1][0], X[0][0], -1)
        LB = add_form(Y[0][0], Y[0][1])
        terms.append((LA, LB))
        
        # s7
        LA = add_form(X[0][1], X[1][1], -1)
        LB = add_form(Y[1][0], Y[1][1])
        terms.append((LA, LB))
        
        return terms

    # Top level Strassen for 4x4
    # A blocks: A11, A12, A21, A22 (each 2x2)
    # B blocks: B11, B12, B21, B22 (each 2x2)
    
    # Initialize block forms
    # A[i][j] is a dict {(row, col): 1}
    def get_A_block(r, c):
        return [[{(r+dr, c+dc): 1} for dc in range(2)] for dr in range(2)]
        
    def get_B_block(r, c):
        return [[{(r+dr, c+dc): 1} for dc in range(2)] for dr in range(2)]

    # We need to construct X_k and Y_k for k=1..7
    # X_k and Y_k are 2x2 matrices of linear forms (dicts)
    
    # P1: X = A11+A22, Y = B11+B22
    # P2: X = A21+A22, Y = B11
    # P3: X = A11, Y = B12-B22
    # P4: X = A22, Y = B21-B11
    # P5: X = A11+A12, Y = B22
    # P6: X = A21-A11, Y = B11+B12
    # P7: X = A12-A22, Y = B21+B22
    
    def add_block(A1, A2, sign=1):
        # A1, A2 are 2x2 lists of dicts
        res = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = add_form(A1[i][j], A2[i][j], sign)
        return res

    A11 = get_A_block(0, 0)
    A12 = get_A_block(0, 2)
    A21 = get_A_block(2, 0)
    A22 = get_A_block(2, 2)
    
    B11 = get_B_block(0, 0)
    B12 = get_B_block(0, 2)
    B21 = get_B_block(2, 0)
    B22 = get_B_block(2, 2)
    
    combos = []
    # P1
    combos.append((add_block(A11, A22), add_block(B11, B22)))
    # P2
    combos.append((add_block(A21, A22), B11))
    # P3
    combos.append((A11, add_block(B12, B22, -1)))
    # P4
    combos.append((A22, add_block(B21, B11, -1)))
    # P5
    combos.append((add_block(A11, A12), B22))
    # P6
    combos.append((add_block(A21, A11, -1), add_block(B11, B12)))
    # P7
    combos.append((add_block(A12, A22, -1), add_block(B21, B22)))
    
    # Process each combo
    # For each combo k, get 7 terms (LA, LB)
    # The result M_k is a 2x2 matrix of linear combinations of these terms
    # M_k[0][0] = T0 - T3 + T4 - T6
    # M_k[0][1] = T2 + T4
    # M_k[1][0] = T1 + T3
    # M_k[1][1] = T0 - T1 + T2 + T5
    
    # We accumulate W coefficients
    # W is 20x49 (for the 4x4 part)
    # We will store the terms in a list `all_terms`
    # each term is (LA, LB, W_contrib_dict)
    # W_contrib_dict maps (i, k) to coeff
    
    all_terms = []
    
    for k, (X, Y) in enumerate(combos):
        terms = strassen_2x2(X, Y)
        # terms is list of 7 (LA, LB)
        # Coefficients for M_k entries
        # M00: T0, -T3, T4, -T6
        # M01: T2, T4
        # M10: T1, T3
        # M11: T0, -T1, T2, T5
        
        coeffs = [
            [(0, 1), (3, -1), (4, 1), (6, -1)], # M00
            [(2, 1), (4, 1)],                   # M01
            [(1, 1), (3, 1)],                   # M10
            [(0, 1), (1, -1), (2, 1), (5, 1)]  # M11
        ]
        
        for t_idx, (LA, LB) in enumerate(terms):
            w_contrib = {}
            for m_idx, coeff_list in enumerate(coeffs):
                for t_j, c in coeff_list:
                    if t_j == t_idx:
                        # m_idx 0 -> C11 (i in 0,1, k in 0,1)
                        # m_idx 1 -> C12 (i in 0,1, k in 2,3)
                        # m_idx 2 -> C21 (i in 2,3, k in 0,1)
                        # m_idx 3 -> C22 (i in 2,3, k in 2,3)
                        
                        if m_idx == 0:
                            for i in range(2):
                                for kk in range(2):
                                    w_contrib[(i, kk)] = w_contrib.get((i, kk), 0) + c
                        elif m_idx == 1:
                            for i in range(2):
                                for kk in range(2, 4):
                                    w_contrib[(i, kk)] = w_contrib.get((i, kk), 0) + c
                        elif m_idx == 2:
                            for i in range(2, 4):
                                for kk in range(2):
                                    w_contrib[(i, kk)] = w_contrib.get((i, kk), 0) + c
                        elif m_idx == 3:
                            for i in range(2, 4):
                                for kk in range(2, 4):
                                    w_contrib[(i, kk)] = w_contrib.get((i, kk), 0) + c
            all_terms.append((LA, LB, w_contrib))
            
    # Add the 16 terms for the last column
    for i in range(4):
        for j in range(4):
            LA = {(i, j): 1}
            LB = {(j, 4): 1}
            w_contrib = {(i, 4): 1}
            all_terms.append((LA, LB, w_contrib))
            
    # Fill U, V, W
    for r, (LA, LB, w_contrib) in enumerate(all_terms):
        for (i, j), c in LA.items():
            U[i * m + j][r] = c
        for (j, k), c in LB.items():
            V[j * p + k][r] = c
        for (k, i), c in w_contrib.items():
            W[k * n + i][r] = c
            
    return {"U": U, "V": V, "W": W}
