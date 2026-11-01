import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 2, 14
    rank = 49  # 7 blocks * 7 Strassen terms per block
    
    # Initialize U, V, W as lists of lists (matching the seed interface)
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # Strassen's 7 products for 2x2 multiplication:
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    
    # U coefficients: [a, b, c, d]
    U_coeff = [
        (1, 0, 0, 1),   # M1
        (0, 0, 1, 1),   # M2
        (1, 0, 0, 0),   # M3
        (0, 0, 0, 1),   # M4
        (1, 1, 0, 0),   # M5
        (-1, 0, 1, 0),  # M6
        (0, 1, 0, -1),  # M7
    ]
    
    # V coefficients: [e, f, g, h]
    V_coeff = [
        (1, 0, 0, 1),   # M1
        (1, 0, 0, 0),   # M2
        (0, 1, 0, -1),  # M3
        (-1, 0, 1, 0),  # M4
        (0, 0, 0, 1),   # M5
        (1, 1, 0, 0),   # M6
        (0, 0, 1, 1),   # M7
    ]
    
    # W coefficients: [C00, C01, C10, C11]
    W_coeff = [
        (1, 0, 0, 1),   # M1
        (0, 0, 1, -1),  # M2
        (0, 1, 0, 1),   # M3
        (1, 0, 1, 0),   # M4
        (-1, 1, 0, 0),  # M5
        (0, 0, 0, 1),   # M6
        (1, 0, 0, 0),   # M7
    ]
    
    for b in range(7):  # 7 blocks
        for s in range(7):  # 7 Strassen terms per block
            col = b * 7 + s
            
            # U: same coefficients for each block
            U[0][col] = U_coeff[s][0]
            U[1][col] = U_coeff[s][1]
            U[2][col] = U_coeff[s][2]
            U[3][col] = U_coeff[s][3]
            
            # V: B row 0 columns (2b, 2b+1), row 1 columns (14+2b, 15+2b)
            row0_c0 = 2 * b
            row0_c1 = 2 * b + 1
            row1_c0 = 14 + 2 * b
            row1_c1 = 14 + 2 * b + 1
            V[row0_c0][col] = V_coeff[s][0]
            V[row0_c1][col] = V_coeff[s][1]
            V[row1_c0][col] = V_coeff[s][2]
            V[row1_c1][col] = V_coeff[s][3]
            
            # W: C = 2 rows x 14 cols, index = k*n + i
            W[4 * b][col] = W_coeff[s][0]       # C[0][2b]
            W[4 * b + 2][col] = W_coeff[s][1]   # C[0][2b+1]
            W[4 * b + 1][col] = W_coeff[s][2]   # C[1][2b]
            W[4 * b + 3][col] = W_coeff[s][3]   # C[1][2b+1]
    
    return {"U": U, "V": V, "W": W}
