def construct():
    n, m, p = 2, 2, 10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # t: 0..6 corresponding to M1..M7
    # U coeffs: (a, b, c, d)
    U_coeffs = [
        (1, 0, 0, 1), # M1
        (0, 0, 1, 1), # M2
        (1, 0, 0, 0), # M3
        (0, 0, 0, 1), # M4
        (1, 1, 0, 0), # M5
        (-1, 0, 1, 0), # M6
        (0, 1, 0, -1)  # M7
    ]
    
    # V coeffs for block l: (e, f, g, h) at indices 2l, 2l+1, 10+2l, 10+2l+1
    # We'll handle this in the loop
    
    # W coeffs for block l: (C00, C10, C01, C11) at indices 4l, 4l+1, 4l+2, 4l+3
    W_coeffs = [
        (1, 0, 0, 1), # M1
        (0, 1, 0, -1), # M2
        (0, 0, 1, 1), # M3
        (1, 1, 0, 0), # M4
        (-1, 0, 1, 0), # M5
        (0, 0, 0, 1), # M6
        (1, 0, 0, 0)  # M7
    ]
    
    for l in range(5):
        for t in range(7):
            c = l * 7 + t
            # Fill U
            u_a, u_b, u_c, u_d = U_coeffs[t]
            U[0][c] = u_a
            U[1][c] = u_b
            U[2][c] = u_c
            U[3][c] = u_d
            
            # Fill V
            # e at 2l, f at 2l+1, g at 10+2l, h at 10+2l+1
            # Determine coeffs for e, f, g, h based on t
            if t == 0: # M1: e+h
                v_e, v_f, v_g, v_h = 1, 0, 0, 1
            elif t == 1: # M2: e
                v_e, v_f, v_g, v_h = 1, 0, 0, 0
            elif t == 2: # M3: f-h
                v_e, v_f, v_g, v_h = 0, 1, 0, -1
            elif t == 3: # M4: g-e
                v_e, v_f, v_g, v_h = -1, 0, 1, 0
            elif t == 4: # M5: h
                v_e, v_f, v_g, v_h = 0, 0, 0, 1
            elif t == 5: # M6: e+f
                v_e, v_f, v_g, v_h = 1, 1, 0, 0
            elif t == 6: # M7: g+h
                v_e, v_f, v_g, v_h = 0, 0, 1, 1
            
            V[2*l][c] = v_e
            V[2*l+1][c] = v_f
            V[10+2*l][c] = v_g
            V[10+2*l+1][c] = v_h
            
            # Fill W
            w_c00, w_c10, w_c01, w_c11 = W_coeffs[t]
            W[4*l][c] = w_c00
            W[4*l+1][c] = w_c10
            W[4*l+2][c] = w_c01
            W[4*l+3][c] = w_c11
            
    return {"U": U, "V": V, "W": W}
