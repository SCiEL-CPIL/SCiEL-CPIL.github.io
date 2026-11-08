"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Total rank: 4 blocks of 2x2x7, each rank 23.
    # 2x2x7 rank 23 comes from 3x(2x2x2 rank 7) + 1x(2x2x1 rank 2)
    # 2x2x2 Strassen rank 7
    # 2x2x1 Schoolbook rank 2
    
    rank_per_block = 23
    total_rank = 4 * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_strassen_2x2x2(u_base, v_base, w_base, col_offset):
        # u_base: list of 4 global U indices for the 2x2 A block
        # v_base: list of 2 global V indices for the 2x2 B block (actually 2 rows of B)
        # w_base: list of 2 global W indices for the 2x2 C block (actually 2 rows of C)
        # Note: B is 2x2 here, so 2 rows. C is 2x2, so 2 rows.
        # Wait, the 2x2x2 block multiplies 2x2 by 2x2.
        # In our context, B is 2x7. We split k into groups.
        # So for a group of 2 k's, B is 2x2.
        # v_base should be the indices for the 2 rows of B (j=0,1) and 2 cols (k=k0, k0+1).
        # So v_base is a list of 4 indices: [j0*k0, j0*k0+1, j1*k0, j1*k0+1] ?
        # No, V index is j*p + k.
        # So for fixed j, k varies.
        
        # Let's define the local indices for the 2x2x2 block.
        # A: 0,1,2,3
        # B: 0,1,2,3 (j=0,1; k=0,1)
        # C: 0,1,2,3 (i=0,1; k=0,1)
        
        # Strassen coefficients
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # We need to add terms to U, V, W.
        # Each term is a rank-1 tensor.
        
        # Term 1: (a+d) * (e+h)
        # U: a=1, d=1. V: e=1, h=1. W: 1 (coeff 1), 2 (coeff -1), 3 (coeff 1), 0 (coeff 1)?
        # C = 
        # [ m1+m4-m5+m7, m3+m5 ]
        # [ m2+m4, m1-m2+m3+m6 ]
        
        # Let's just generate the 7 rank-1 terms for Strassen.
        # The standard Strassen decomposition is:
        # m1 = (a+d)(e+h) -> U: a,d. V: e,h. W: p,s (coeffs 1,1), q,r? No.
        # p = m1 + m4 - m5 + m7
        # q = m3 + m5
        # r = m2 + m4
        # s = m1 - m2 + m3 + m6
        
        # This requires distributing the m's to the outputs.
        # It is easier to define the 7 bilinear forms and their contributions to C.
        
        # Form 1: (a+d)(e+h)
        # Contributes to: p (+1), s (+1)
        # Form 2: (c+d)e
        # Contributes to: r (+1), s (-1)
        # Form 3: a(f-h)
        # Contributes to: q (+1), s (+1)
        # Form 4: d(g-e)
        # Contributes to: p (+1), r (+1)
        # Form 5: (a+b)h
        # Contributes to: p (-1), q (+1)
        # Form 6: (c-a)(e+f)
        # Contributes to: s (+1)
        # Form 7: (b-d)(g+h)
        # Contributes to: p (+1)
        
        # Let's verify:
        # p = m1 + m4 - m5 + m7. Correct.
        # q = m3 + m5. Correct.
        # r = m2 + m4. Correct.
        # s = m1 - m2 + m3 + m6. Correct.
        
        # Now map to indices.
        # A indices: 0(a), 1(b), 2(c), 3(d)
        # B indices: 0(e), 1(f), 2(g), 3(h)
        # C indices: 0(p), 1(q), 2(r), 3(s)
        
        terms = [
            # (coeff_U, coeff_V, coeff_W)
            # We need to add these to the global U, V, W matrices.
            # Each term is a product of a linear form in A, linear form in B, linear form in C.
            # We add to U[row][col] += coeff_U
            # V[row][col] += coeff_V
            # W[row][col] += coeff_W
            
            # Term 1: (a+d)(e+h)
            # U: a(1), d(1). V: e(1), h(1). W: p(1), s(1).
            # This is not a single rank-1 term in the decomposition sense?
            # Yes, it is. The rank-1 tensor is $u \otimes v \otimes w$.
            # $u = e_a + e_d$. $v = e_e + e_h$. $w = e_p + e_s$.
            # So we add:
            # U[u_a][col] += 1, U[u_d][col] += 1
            # V[v_e][col] += 1, V[v_h][col] += 1
            # W[w_p][col] += 1, W[w_s][col] += 1
            # This is valid.
            
            # Let's list all 7.
        ]
        
        # Helper to add a term
        def add_term(u_coeffs, v_coeffs, w_coeffs, col):
            # u_coeffs: dict {local_idx: coeff}
            # v_coeffs: dict {local_idx: coeff}
            # w_coeffs: dict {local_idx: coeff}
            for li, c in u_coeffs.items():
                U[u_base[li]][col] += c
            for li, c in v_coeffs.items():
                V[v_base[li]][col] += c
            for li, c in w_coeffs.items():
                W[w_base[li]][col] += c

        # Term 1: (a+d)(e+h) -> p, s
        add_term({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}, col_offset)
        
        # Term 2: (c+d)e -> r, s(-1)
        add_term({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}, col_offset + 1)
        
        # Term 3: a(f-h) -> q, s
        add_term({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}, col_offset + 2)
        
        # Term 4: d(g-e) -> p, r
        add_term({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}, col_offset + 3)
        
        # Term 5: (a+b)h -> p(-1), q
        add_term({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}, col_offset + 4)
        
        # Term 6: (c-a)(e+f) -> s
        add_term({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}, col_offset + 5)
        
        # Term 7: (b-d)(g+h) -> p
        add_term({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}, col_offset + 6)

    def add_schoolbook_2x2x1(u_base, v_base, w_base, col_offset):
        # Multiplies 2x2 by 2x1. Result 2x1.
        # C0 = a*e + c*g
        # C1 = b*e + d*g
        # Term 1: a*e -> C0
        # Term 2: c*g -> C0
        # Term 3: b*e -> C1
        # Term 4: d*g -> C1
        
        def add_term(u_coeffs, v_coeffs, w_coeffs, col):
            for li, c in u_coeffs.items():
                U[u_base[li]][col] += c
            for li, c in v_coeffs.items():
                V[v_base[li]][col] += c
            for li, c in w_coeffs.items():
                W[w_base[li]][col] += c

        # A: 0(a), 1(b), 2(c), 3(d)
        # B: 0(e), 1(g)  (Note: B is 2x1, so only col 0. j=0,1)
        # C: 0(C0), 1(C1)
        
        add_term({0: 1}, {0: 1}, {0: 1}, col_offset)
        add_term({2: 1}, {1: 1}, {0: 1}, col_offset + 1)
        add_term({1: 1}, {0: 1}, {1: 1}, col_offset + 2)
        add_term({3: 1}, {1: 1}, {1: 1}, col_offset + 3)

    # Block 1: A[0:2, 0:2] * B[0:2, :]
    # A indices: 0,1,4,5
    # B indices: 0..6, 7..13
    # C indices: 0,1, 4,5, 8,9, 12,13, 16,17, 20,21, 24,25
    
    # We split B columns (k) into 0,1; 2,3; 4,5; 6
    # Block 1 uses B rows 0,1.
    # u_base1 = [0, 1, 4, 5]
    # For k=0,1: v_base = [0, 1, 7, 8] (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1)
    #            w_base = [0, 1, 4, 5] (i=0,k=0; i=0,k=1; i=1,k=0; i=1,k=1)
    # For k=2,3: v_base = [2, 3, 9, 10]
    #            w_base = [8, 9, 12, 13]
    # For k=4,5: v_base = [4, 5, 11, 12]
    #            w_base = [16, 17, 20, 21]
    # For k=6:   v_base = [6, 13]
    #            w_base = [24, 25]
    
    u_b1 = [0, 1, 4, 5]
    
    # k=0,1
    v_b1_0 = [0, 1, 7, 8]
    w_b1_0 = [0, 1, 4, 5]
    add_strassen_2x2x2(u_b1, v_b1_0, w_b1_0, 0)
    
    # k=2,3
    v_b1_1 = [2, 3, 9, 10]
    w_b1_1 = [8, 9, 12, 13]
    add_strassen_2x2x2(u_b1, v_b1_1, w_b1_1, 7)
    
    # k=4,5
    v_b1_2 = [4, 5, 11, 12]
    w_b1_2 = [16, 17, 20, 21]
    add_strassen_2x2x2(u_b1, v_b1_2, w_b1_2, 14)
    
    # k=6
    v_b1_3 = [6, 13]
    w_b1_3 = [24, 25]
    add_schoolbook_2x2x1(u_b1, v_b1_3, w_b1_3, 21)
    
    # Block 2: A[0:2, 2:4] * B[2:4, :]
    # A indices: 2,3,6,7
    # B indices: 14..20, 21..27
    # C indices: 0,1, 4,5, 8,9, 12,13, 16,17, 20,21, 24,25
    
    u_b2 = [2, 3, 6, 7]
    
    # k=0,1
    v_b2_0 = [14, 15, 21, 22]
    w_b2_0 = [0, 1, 4, 5]
    add_strassen_2x2x2(u_b2, v_b2_0, w_b2_0, 23)
    
    # k=2,3
    v_b2_1 = [16, 17, 23, 24]
    w_b2_1 = [8, 9, 12, 13]
    add_strassen_2x2x2(u_b2, v_b2_1, w_b2_1, 30)
    
    # k=4,5
    v_b2_2 = [18, 19, 25, 26]
    w_b2_2 = [16, 17, 20, 21]
    add_strassen_2x2x2(u_b2, v_b2_2, w_b2_2, 37)
    
    # k=6
    v_b2_3 = [20, 27]
    w_b2_3 = [24, 25]
    add_schoolbook_2x2x1(u_b2, v_b2_3, w_b2_3, 44)
    
    # Block 3: A[2:4, 0:2] * B[0:2, :]
    # A indices: 8,9,12,13
    # B indices: 0..6, 7..13
    # C indices: 16,17, 20,21, 24,25, 28,29, 32,33, 36,37, 40,41
    
    u_b3 = [8, 9, 12, 13]
    
    # k=0,1
    v_b3_0 = [0, 1, 7, 8]
    w_b3_0 = [16, 17, 20, 21]
    add_strassen_2x2x2(u_b3, v_b3_0, w_b3_0, 46)
    
    # k=2,3
    v_b3_1 = [2, 3, 9, 10]
    w_b3_1 = [24, 25, 28, 29]
    add_strassen_2x2x2(u_b3, v_b3_1, w_b3_1, 53)
    
    # k=4,5
    v_b3_2 = [4, 5, 11, 12]
    w_b3_2 = [32, 33, 36, 37]
    add_strassen_2x2x2(u_b3, v_b3_2, w_b3_2, 60)
    
    # k=6
    v_b3_3 = [6, 13]
    w_b3_3 = [40, 41]
    add_schoolbook_2x2x1(u_b3, v_b3_3, w_b3_3, 67)
    
    # Block 4: A[2:4, 2:4] * B[2:4, :]
    # A indices: 10,11,14,15
    # B indices: 14..20, 21..27
    # C indices: 16,17, 20,21, 24,25, 28,29, 32,33, 36,37, 40,41
    
    u_b4 = [10, 11, 14, 15]
    
    # k=0,1
    v_b4_0 = [14, 15, 21, 22]
    w_b4_0 = [16, 17, 20, 21]
    add_strassen_2x2x2(u_b4, v_b4_0, w_b4_0, 69)
    
    # k=2,3
    v_b4_1 = [16, 17, 23, 24]
    w_b4_1 = [24, 25, 28, 29]
    add_strassen_2x2x2(u_b4, v_b4_1, w_b4_1, 76)
    
    # k=4,5
    v_b4_2 = [18, 19, 25, 26]
    w_b4_2 = [32, 33, 36, 37]
    add_strassen_2x2x2(u_b4, v_b4_2, w_b4_2, 83)
    
    # k=6
    v_b4_3 = [20, 27]
    w_b4_3 = [40, 41]
    add_schoolbook_2x2x1(u_b4, v_b4_3, w_b4_3, 90)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
