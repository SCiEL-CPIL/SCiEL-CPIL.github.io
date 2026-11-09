"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 56 (4 blocks * 2 sources * 7 Strassen mults)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A part)
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # M0: a+d -> [1, 0, 0, 1]
    # M1: c+d -> [0, 0, 1, 1]
    # M2: a   -> [1, 0, 0, 0]
    # M3: d   -> [0, 0, 0, 1]
    # M4: a+b -> [1, 1, 0, 0]
    # M5: c-d -> [0, 0, 1, -1]
    # M6: b-d -> [0, 1, 0, -1]
    U_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    
    # Strassen coefficients for V (B part)
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # M0: e+h -> [1, 0, 0, 1]
    # M1: e   -> [1, 0, 0, 0]
    # M2: f-h -> [0, 1, 0, -1]
    # M3: g-e -> [0, 0, 1, -1]
    # M4: h   -> [0, 0, 0, 1]
    # M5: g+h -> [0, 0, 1, 1]
    # M6: e+f -> [1, 1, 0, 0]
    V_coeffs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ]
    
    # Strassen coefficients for W (C part)
    # z00, z01, z10, z11
    # M0: +1,  0,  0, +1
    # M1:  0,  0, +1, -1
    # M2:  0, +1,  0, +1
    # M3: +1,  0, +1,  0
    # M4: -1, +1,  0,  0
    # M5:  0,  0,  0, +1
    # M6: +1,  0,  0,  0
    W_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    # Iterate over 2 sources (A1, A2) and 4 blocks (k-pairs)
    for source in range(2):
        # source 0: A1 (cols 0,1 of A), B1 (rows 0,1 of B)
        # source 1: A2 (cols 2,3 of A), B2 (rows 2,3 of B)
        a_col_offset = source * 2
        b_row_offset = source * 2
        
        for t in range(4):
            k_offset = t * 2
            
            for m_idx in range(7):
                # U: A part
                # A indices: i in {0,1}, j in {0,1} local -> global j = j_local + a_col_offset
                for i in range(2):
                    for j_local in range(2):
                        val = U_coeffs[m_idx][i*2 + j_local]
                        if val != 0:
                            U[i * m + (j_local + a_col_offset)][col] = val
                
                # V: B part
                # B indices: j in {0,1} local -> global j = j_local + b_row_offset
                #           k in {0,1} local -> global k = k_local + k_offset
                for j_local in range(2):
                    for k_local in range(2):
                        val = V_coeffs[m_idx][j_local*2 + k_local]
                        if val != 0:
                            V[(j_local + b_row_offset) * p + (k_local + k_offset)][col] = val
                
                # W: C part
                # C indices: i in {0,1}, k in {0,1} local -> global k = k_local + k_offset
                # W index convention: W[k*n + i, r]
                for i in range(2):
                    for k_local in range(2):
                        val = W_coeffs[m_idx][i*2 + k_local]
                        if val != 0:
                            W[(k_local + k_offset) * n + i][col] = val
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
