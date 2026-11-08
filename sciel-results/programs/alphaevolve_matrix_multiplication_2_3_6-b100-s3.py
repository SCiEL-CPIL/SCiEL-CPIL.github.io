"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (6, R)
    # V shape: (18, R)
    # W shape: (12, R)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # We will build the terms and then transpose to U, V, W
    # Each term is a tuple (u_vec, v_vec, w_vec)
    terms = []
    
    # Helper to create a vector with specific values
    def make_vec(size, indices_vals):
        vec = [0] * size
        for idx, val in indices_vals.items():
            vec[idx] = val
        return vec

    # Part 1: A_12 * B_2
    # A_12 is col 2 of A (indices 2, 5 in U)
    # B_2 is row 2 of B (indices 12..17 in V)
    # Output C is 2x6 (indices 0..11 in W)
    # Term: A_i2 * B_2k
    for i in range(2):
        for k in range(6):
            u_idx = i * 3 + 2 # A[i][2]
            v_idx = 2 * 6 + k # B[2][k]
            w_idx = k * 2 + i # C[k][i]
            
            u_vec = make_vec(6, {u_idx: 1})
            v_vec = make_vec(18, {v_idx: 1})
            w_vec = make_vec(12, {w_idx: 1})
            terms.append((u_vec, v_vec, w_vec))
            
    # Part 2: A_11 * B_1
    # A_11 is cols 0,1 of A
    # B_1 is rows 0,1 of B, split into 3 blocks of 2 cols each
    # Block j of B_1 corresponds to cols 2*j, 2*j+1 of B
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A_11 (2x2):
    # A[0][0] -> U idx 0
    # A[0][1] -> U idx 1
    # A[1][0] -> U idx 3
    # A[1][1] -> U idx 4
    
    # For each block j (0, 1, 2) of B_1
    for j in range(3):
        # B block columns: 2*j, 2*j+1
        # B[0][2j] -> V idx 0*6 + 2j = 2j
        # B[0][2j+1] -> V idx 0*6 + 2j+1 = 2j+1
        # B[1][2j] -> V idx 1*6 + 2j = 6+2j
        # B[1][2j+1] -> V idx 1*6 + 2j+1 = 7+2j
        
        b00_idx = 2*j
        b01_idx = 2*j + 1
        b10_idx = 6 + 2*j
        b11_idx = 6 + 2*j + 1
        
        # Output C block for this j:
        # C[0][2j], C[0][2j+1]
        # C[1][2j], C[1][2j+1]
        # W indices:
        # C[0][2j] -> 2j * 2 + 0 = 4j
        # C[0][2j+1] -> (2j+1)*2 + 0 = 4j+2
        # C[1][2j] -> 2j * 2 + 1 = 4j+1
        # C[1][2j+1] -> (2j+1)*2 + 1 = 4j+3
        
        c00_idx = 4*j
        c01_idx = 4*j + 2
        c10_idx = 4*j + 1
        c11_idx = 4*j + 3
        
        # Define the 7 Strassen terms
        # Each term: (u_indices, v_indices, w_coefficients)
        
        # M1
        u1 = {0: 1, 4: 1} # A00, A11
        v1 = {b00_idx: 1, b11_idx: 1} # B00, B11
        w1 = {c00_idx: 1, c11_idx: 1}
        
        # M2
        u2 = {3: 1, 4: 1} # A10, A11
        v2 = {b00_idx: 1} # B00
        w2 = {c10_idx: 1, c11_idx: -1}
        
        # M3
        u3 = {0: 1} # A00
        v3 = {b01_idx: 1, b11_idx: -1} # B01, -B11
        w3 = {c01_idx: 1, c11_idx: 1}
        
        # M4
        u4 = {4: 1} # A11
        v4 = {b10_idx: 1, b00_idx: -1} # B10, -B00
        w4 = {c00_idx: 1, c10_idx: 1}
        
        # M5
        u5 = {0: 1, 1: 1} # A00, A01
        v5 = {b11_idx: 1} # B11
        w5 = {c00_idx: -1, c01_idx: 1}
        
        # M6
        u6 = {3: 1, 0: -1} # A10, -A00
        v6 = {b00_idx: 1, b01_idx: 1} # B00, B01
        w6 = {c11_idx: 1}
        
        # M7
        u7 = {1: 1, 4: -1} # A01, -A11
        v7 = {b10_idx: 1, b11_idx: 1} # B10, B11
        w7 = {c00_idx: 1}
        
        # Add terms
        terms.append((make_vec(6, u1), make_vec(18, v1), make_vec(12, w1)))
        terms.append((make_vec(6, u2), make_vec(18, v2), make_vec(12, w2)))
        terms.append((make_vec(6, u3), make_vec(18, v3), make_vec(12, w3)))
        terms.append((make_vec(6, u4), make_vec(18, v4), make_vec(12, w4)))
        terms.append((make_vec(6, u5), make_vec(18, v5), make_vec(12, w5)))
        terms.append((make_vec(6, u6), make_vec(18, v6), make_vec(12, w6)))
        terms.append((make_vec(6, u7), make_vec(18, v7), make_vec(12, w7)))
        
    # Build U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(12)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(6):
            U[i][c] = u[i]
        for i in range(18):
            V[i][c] = v[i]
        for i in range(12):
            W[i][c] = w[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
