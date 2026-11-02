"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm coefficients for 2x2 matrix multiplication
    # Each term is (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict {local_idx: coeff}
    # b_coeffs: dict {local_idx: coeff}
    # c_coeffs: dict {local_idx: coeff}
    
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # M1 = (A00+A11)(B00+B11)
    m1_a = {0: 1, 3: 1}
    m1_b = {0: 1, 3: 1}
    m1_c = {0: 1, 3: 1} # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    m2_a = {2: 1, 3: 1}
    m2_b = {0: 1}
    m2_c = {2: 1, 3: -1} # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    m3_a = {0: 1}
    m3_b = {1: 1, 3: -1}
    m3_c = {1: 1, 3: 1} # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    m4_a = {3: 1}
    m4_b = {2: 1, 0: -1}
    m4_c = {0: 1, 2: 1} # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    m5_a = {0: 1, 1: 1}
    m5_b = {3: 1}
    m5_c = {0: -1, 1: 1} # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    m6_a = {2: 1, 0: -1}
    m6_b = {0: 1, 1: 1}
    m6_c = {3: 1} # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    m7_a = {1: 1, 3: -1}
    m7_b = {2: 1, 3: 1}
    m7_c = {0: 1} # C00 += M7
    
    terms = [
        (m1_a, m1_b, m1_c),
        (m2_a, m2_b, m2_c),
        (m3_a, m3_b, m3_c),
        (m4_a, m4_b, m4_c),
        (m5_a, m5_b, m5_c),
        (m6_a, m6_b, m6_c),
        (m7_a, m7_b, m7_c)
    ]
    
    # Define the 4 block multiplications
    # Each block is defined by offsets for A, B, C
    # A offset: (i_off, j_off)
    # B offset: (j_off, k_off)
    # C offset: (k_off, i_off)
    
    blocks = [
        # Block 1: A1 B11
        {'a_off': (0, 0), 'b_off': (0, 0), 'c_off': (0, 0)},
        # Block 2: A2 B21
        {'a_off': (0, 2), 'b_off': (2, 0), 'c_off': (0, 0)},
        # Block 3: A1 B12
        {'a_off': (0, 0), 'b_off': (0, 2), 'c_off': (2, 0)},
        # Block 4: A2 B22
        {'a_off': (0, 2), 'b_off': (2, 2), 'c_off': (2, 0)}
    ]
    
    col_idx = 0
    for block in blocks:
        ai, aj = block['a_off']
        bj, bk = block['b_off']
        ck, ci = block['c_off']
        
        for a_coeffs, b_coeffs, c_coeffs in terms:
            for local_a, coeff in a_coeffs.items():
                li, lj = divmod(local_a, 2)
                gi, gj = ai + li, aj + lj
                U[gi * m + gj][col_idx] += coeff
            
            for local_b, coeff in b_coeffs.items():
                lj, lk = divmod(local_b, 2)
                gj, gk = bj + lj, bk + lk
                V[gj * p + gk][col_idx] += coeff
            
            for local_c, coeff in c_coeffs.items():
                lk, li = divmod(local_c, 2)
                gk, gi = ck + lk, ci + li
                W[gk * n + gi][col_idx] += coeff
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
