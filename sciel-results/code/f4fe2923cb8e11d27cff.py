import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # u_coefs[t][i][j]
    u_coefs = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1, (1,1): 1 }, # M2
        { (0,0): 1 },           # M3
        { (1,1): 1 },           # M4
        { (0,0): 1, (0,1): 1 }, # M5
        { (1,0): 1, (0,0): -1 },# M6
        { (0,1): 1, (1,1): -1 } # M7
    ]
    
    # v_coefs[t][j][k_local] where k_local is 0 or 1
    v_coefs = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (0,0): 1 },           # M2
        { (0,1): 1, (1,1): -1 },# M3
        { (1,0): 1, (0,0): -1 },# M4
        { (1,1): 1 },           # M5
        { (0,0): 1, (0,1): 1 }, # M6
        { (1,0): 1, (1,1): 1 }  # M7
    ]
    
    # w_coefs[t][i][k_local]
    w_coefs = [
        { (0,0): 1, (1,1): 1 }, # M1 -> C00, C11
        { (1,0): 1, (1,1): -1 },# M2 -> C10, -C11
        { (0,1): 1, (1,1): 1 }, # M3 -> C01, C11
        { (0,0): 1, (1,0): 1 }, # M4 -> C00, C10
        { (0,0): -1, (0,1): 1 },# M5 -> -C00, C01
        { (1,1): 1 },           # M6 -> C11
        { (0,0): 1 }            # M7 -> C00
    ]
    
    for s in range(3):
        base_col = 7 * s
        for t in range(7):
            col = base_col + t
            
            # Fill U
            for (i, j), val in u_coefs[t].items():
                row_u = i * m + j
                U[row_u][col] += val
                
            # Fill V
            for (j, k_loc), val in v_coefs[t].items():
                k_global = 2 * s + k_loc
                row_v = j * p + k_global
                V[row_v][col] += val
                
            # Fill W
            for (i, k_loc), val in w_coefs[t].items():
                k_global = 2 * s + k_loc
                row_w = k_global * n + i
                W[row_w][col] += val
                
    return {"U": U, "V": V, "W": W}
