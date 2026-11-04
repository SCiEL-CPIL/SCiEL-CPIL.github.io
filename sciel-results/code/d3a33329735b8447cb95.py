"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition:
    # Split m=3 into m1=2 and m2=1.
    # Part 1: 2x2x10 using 5 Strassen blocks of 2x2x2. Rank 35.
    # Part 2: 2x1x10 using schoolbook. Rank 20.
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for 2x2x10
    # j indices 0, 1
    # k blocks: [0,1], [2,3], [4,5], [6,7], [8,9]
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for 2x2x2
        # M1: (A00+A11)(B00+B11) -> C00, C11
        # M2: (A10+A11)B00 -> C10
        # M3: A00(B01-B11) -> C01
        # M4: A11(B10-B00) -> C00, C10
        # M5: (A00+A01)B11 -> -C00, C01
        # M6: (A10-A00)(B00+B01) -> C11
        # M7: (A01-A11)(B10+B11) -> C00, C11
        
        # Helper to add a term
        def add_term(U_coeffs, V_coeffs, W_coeffs):
            nonlocal col
            for idx, val in U_coeffs.items():
                U[idx][col] += val
            for idx, val in V_coeffs.items():
                V[idx][col] += val
            for idx, val in W_coeffs.items():
                W[idx][col] += val
            col += 1
        
        # Global indices:
        # U row: i*m + j (i in 0..1, j in 0..1)
        # V row: j*p + k (j in 0..1, k in k0,k1)
        # W row: k*n + i (k in k0,k1, i in 0..1)
        
        # M1
        U1 = {0*3+0: 1, 1*3+1: 1} # A00, A11
        V1 = {0*10+k0: 1, 1*10+k1: 1} # B00, B11
        W1 = {k0*2+0: 1, k1*2+1: 1} # C00, C11
        add_term(U1, V1, W1)
        
        # M2
        U2 = {1*3+0: 1, 1*3+1: 1} # A10, A11
        V2 = {0*10+k0: 1} # B00
        W2 = {k0*2+1: 1} # C10
        add_term(U2, V2, W2)
        
        # M3
        U3 = {0*3+0: 1} # A00
        V3 = {0*10+k1: 1, 1*10+k1: -1} # B01 - B11
        W3 = {k1*2+0: 1} # C01
        add_term(U3, V3, W3)
        
        # M4
        U4 = {1*3+1: 1} # A11
        V4 = {1*10+k0: 1, 0*10+k0: -1} # B10 - B00
        W4 = {k0*2+0: 1, k0*2+1: 1} # C00, C10
        add_term(U4, V4, W4)
        
        # M5
        U5 = {0*3+0: 1, 0*3+1: 1} # A00, A01
        V5 = {1*10+k1: 1} # B11
        W5 = {k0*2+0: -1, k1*2+0: 1} # -C00, C01
        add_term(U5, V5, W5)
        
        # M6
        U6 = {1*3+0: 1, 0*3+0: -1} # A10 - A00
        V6 = {0*10+k0: 1, 0*10+k1: 1} # B00, B01
        W6 = {k1*2+1: 1} # C11
        add_term(U6, V6, W6)
        
        # M7
        U7 = {0*3+1: 1, 1*3+1: -1} # A01 - A11
        V7 = {1*10+k0: 1, 1*10+k1: 1} # B10, B11
        W7 = {k0*2+0: 1, k1*2+1: 1} # C00, C11
        add_term(U7, V7, W7)
        
    # Part 2: Schoolbook for 2x1x10
    # j index 2
    for i in range(2):
        for k in range(10):
            U[i*3 + 2][col] = 1
            V[2*10 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
