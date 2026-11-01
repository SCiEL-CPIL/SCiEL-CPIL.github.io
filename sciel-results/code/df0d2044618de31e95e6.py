"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14 is the minimal rank for 2x2x4 matrix multiplication
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # U forms (rows 0..3 correspond to a,b,c,d)
    u_forms = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    # V forms for B1 (cols 0,1) and B2 (cols 2,3)
    # V rows 0..3 correspond to e,f,g,h of B1
    # V rows 4..7 correspond to i,j,k,l of B2
    v_forms_block1 = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [0, 0, 1, -1],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # W forms (coefficients for M1..M7) for each output element
    # C00, C01, C10, C11
    w_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],  # C00: M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],   # C01: M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10: M2 + M4
        [1, -1, 1, 0, 0, 1, 0]   # C11: M1 - M2 + M3 + M6
    ]
    
    # Fill U, V, W for Block 1 (cols 0,1) -> r = 0..6
    for r in range(7):
        for i in range(n*m):
            U[i][r] = u_forms[r][i]
        
        # V for B1 (rows 0..3)
        for j in range(m*p//2):
            V[j][r] = v_forms_block1[r][j]
        # V for B2 (rows 4..7) is 0
        
        # W for C[:, 0:2]
        # Indices: k*n+i
        # C00: k=0, i=0 -> idx 0
        # C01: k=1, i=0 -> idx 2
        # C10: k=0, i=1 -> idx 1
        # C11: k=1, i=1 -> idx 3
        W[0][r] = w_coeffs[0][r]
        W[2][r] = w_coeffs[1][r]
        W[1][r] = w_coeffs[2][r]
        W[3][r] = w_coeffs[3][r]
        # W for C[:, 2:4] is 0
        
    # Fill U, V, W for Block 2 (cols 2,3) -> r = 7..13
    for r in range(7):
        r_offset = 7 + r
        for i in range(n*m):
            U[i][r_offset] = u_forms[r][i]
        
        # V for B1 (rows 0..3) is 0
        # V for B2 (rows 4..7)
        for j in range(m*p//2):
            V[4 + j][r_offset] = v_forms_block1[r][j]
            
        # W for C[:, 2:4]
        # C02: k=2, i=0 -> idx 4
        # C03: k=3, i=0 -> idx 6
        # C12: k=2, i=1 -> idx 5
        # C13: k=3, i=1 -> idx 7
        W[4][r_offset] = w_coeffs[0][r]
        W[6][r_offset] = w_coeffs[1][r]
        W[5][r_offset] = w_coeffs[2][r]
        W[7][r_offset] = w_coeffs[3][r]
        # W for C[:, 0:2] is 0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
