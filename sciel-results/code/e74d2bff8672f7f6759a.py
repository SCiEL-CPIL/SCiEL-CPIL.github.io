def construct():
    n, m, p = 2, 2, 16
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for P1..P7
    # U coeffs: [A00, A01, A10, A11]
    # V coeffs: [B00, B01, B10, B11]
    # W coeffs: [C00, C10, C01, C11]
    
    P = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # P3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]), # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # P5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # P7
    ]
    
    for r in range(8):
        for s in range(7):
            c = r * 7 + s
            u_coeff, v_coeff, w_coeff = P[s]
            
            # Fill U
            for i in range(4):
                U[i][c] = u_coeff[i]
                
            # Fill V
            v_rows = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for i in range(4):
                V[v_rows[i]][c] = v_coeff[i]
                
            # Fill W
            w_rows = [4*r, 4*r+1, 4*r+2, 4*r+3]
            for i in range(4):
                W[w_rows[i]][c] = w_coeff[i]
                
    return {"U": U, "V": V, "W": W}
