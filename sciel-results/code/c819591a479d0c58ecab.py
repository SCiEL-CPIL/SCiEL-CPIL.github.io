import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 6
    
    # Strassen coefficients for 2x2x2
    # Each term is (A_coefs, B_coefs, C_coefs)
    # Keys are (row, col) in 2x2
    strassen_terms = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}), # Wait, C00 has +M1, C11 has +M1?
        # Let's re-verify C formula.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # So M1 contributes to C00 (+1) and C11 (+1).
        # M2 contributes to C10 (+1) and C11 (-1).
        # M3 contributes to C01 (+1) and C11 (+1).
        # M4 contributes to C00 (+1) and C10 (+1).
        # M5 contributes to C00 (-1) and C01 (+1).
        # M6 contributes to C11 (-1).
        # M7 contributes to C00 (+1).
        
        # Let's list terms properly.
    ]
    
    # M1: A=(00,11), B=(00,11), C=(00, 11) -> Coeffs: C00:+1, C11:+1
    # M2: A=(10,11), B=(00), C=(10, 11) -> Coeffs: C10:+1, C11:-1
    # M3: A=(00), B=(10, 11), C=(01, 11) -> Coeffs: C01:+1, C11:+1 (Note B is (10)- (11))
    # M4: A=(11), B=(01, 00), C=(00, 10) -> Coeffs: C00:+1, C10:+1 (Note B is (01)- (00))
    # M5: A=(00, 01), B=(11), C=(00, 01) -> Coeffs: C00:-1, C01:+1
    # M6: A=(10, 00), B=(00, 01), C=(11) -> Coeffs: C11:-1 (Note A is (10)- (00))
    # M7: A=(01, 11), B=(10, 11), C=(00) -> Coeffs: C00:+1 (Note A is (01)- (11))
    
    terms = [
        # (A_coefs, B_coefs, C_coefs)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # M2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7
    ]

    rank = 12 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(2):
        for c in range(2):
            for b in range(3):
                for a_coefs, b_coefs, c_coefs in terms:
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a + ar
                        gj = 2*c + ac
                        U[gi * m + gj][col] += coef
                    for (br, bc), coef in b_coefs.items():
                        gi = 2*c + br
                        gj = 2*b + bc
                        V[gi * p + gj][col] += coef
                    for (cr, cc), coef in c_coefs.items():
                        gi = 2*b + cr
                        gj = 2*a + cc
                        W[gi * n + gj][col] += coef
                    col += 1
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if something went wrong
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col] = 1
                    V_s[j * p + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
