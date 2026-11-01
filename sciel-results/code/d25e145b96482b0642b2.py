"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    rank = 35  # 5 blocks * 7 Strassen factors
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen's 7 bilinear forms for 2x2 x 2x2
    # A indices: A00=0, A01=1, A10=2, A11=3
    # For each block b (0..4), B columns are 2b, 2b+1
    # V indices: B[0][c] = c, B[1][c] = 10+c
    # W indices: C[i][k] -> W[k*2+i]
    #   C[0][2b] -> W[4b], C[0][2b+1] -> W[4b+1]
    #   C[1][2b] -> W[4b+2], C[1][2b+1] -> W[4b+3]
    
    # Strassen factors: (U_coeffs, V_entries, W_entries)
    # U_coeffs: dict mapping A-index to coefficient
    # V_entries: list of (V_index, coeff)
    # W_entries: list of (W_index, coeff)
    
    strassen_factors = [
        # M1 = (A00+A11)(B00+B11) -> C00: +1, C11: +1
        ({0: 1, 3: 1}, [(0, 1), (1, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)B00 -> C10: +1
        ({2: 1, 3: 1}, [(0, 1)], [(2, 1)]),
        # M3 = A00(B01-B11) -> C01: +1
        ({0: 1}, [(1, 1), (11, -1)], [(1, 1)]),
        # M4 = A11(B10-B00) -> C00: +1, C10: +1
        ({3: 1}, [(10, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00+A01)B11 -> C00: -1, C01: +1
        ({0: 1, 1: 1}, [(11, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10-A00)(B00+B01) -> C11: +1
        ({2: 1, 0: -1}, [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01-A11)(B10+B11) -> C00: +1, C11: +1
        ({1: 1, 3: -1}, [(10, 1), (11, 1)], [(0, 1), (3, 1)]),
    ]
    
    for b in range(5):
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_entries, w_entries = strassen_factors[s]
            
            # Fill U
            for a_idx, coeff in u_coeffs.items():
                U[a_idx][col] = coeff
            
            # Fill V
            for v_idx, coeff in v_entries:
                V[v_idx][col] = coeff
            
            # Fill W
            for w_idx, coeff in w_entries:
                W[4 * b + w_idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
