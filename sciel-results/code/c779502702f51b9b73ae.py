"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 50.
    # Split p=7 into p1=4 and p2=3.
    # M_{2,4,7} = M_{2,4,4} + M_{2,4,3}
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        # u_idx is index in U (n*m), v_idx in V (m*p), w_idx in W (p*n)
        # This function appends a column to U, V, W
        # We store as lists of tuples (idx, val) to build sparse matrices later
        pass

    # Helper to add a rank-1 term
    def add_rank1(u_row, v_row, w_row, u_val, v_val, w_val):
        U_cols.append((u_row, u_val))
        V_cols.append((v_row, v_val))
        W_cols.append((w_row, w_val))

    # --- Part 1: M_{2, 4, 4} (Rank 28) ---
    # A is 2x4, B is 4x4.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3).
    # Split B into B1 (rows 0,1) and B2 (rows 2,3).
    # C = A1 B1 + A2 B2.
    # A1 B1 is M_{2,2,4}. A2 B2 is M_{2,2,4}.
    
    # For M_{2,2,4}:
    # A is 2x2, B is 2x4.
    # Split B into B_left (cols 0,1) and B_right (cols 2,3).
    # C = A B_left + A B_right.
    # A B_left is M_{2,2,2} (Strassen, rank 7).
    # A B_right is M_{2,2,2} (Strassen, rank 7).
    
    # Strassen for M_{2,2,2}:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    def add_strassen(a_offset, b_offset, c_offset, d_offset, e_offset, f_offset, g_offset, h_offset, base_u, base_v, base_w):
        # a_offset: index offset for A element (0..3) in local 2x2
        # We map local indices to global indices.
        # Local A is 2x2. Global A is 2x4.
        # Local B is 2x2. Global B is 4x4 (or 4x3).
        
        # Terms:
        # 1. (a+d)(e+h)
        # U: a, d. V: e, h. W: c11, c22
        add_rank1(base_u + a_offset, base_v + e_offset, base_w + 0, 1, 1, 1) # c11 part? No, W index is k*n+i.
        # Let's use a generic Strassen adder that takes mapping functions.
        pass

    # This is getting complex to map indices manually.
    # I will use a simpler approach: build the full 56 rank tensor and then merge?
    # No, I will explicitly list the 50 terms.
    
    # Global indices:
    # U: i*m + j (i in 0..1, j in 0..3)
    # V: j*p + k (j in 0..3, k in 0..6)
    # W: k*n + i (k in 0..6, i in 0..1)
    
    # Part 1: M_{2,4,4} (k in 0..3)
    # A1 (j=0,1) * B1 (j=0,1, k=0..3) -> M_{2,2,4}
    # A2 (j=2,3) * B2 (j=2,3, k=0..3) -> M_{2,2,4}
    
    # M_{2,2,4} for A1 (j=0,1) and B1 (j=0,1, k=0..3)
    # B1 is 2x4. Split into B1_left (k=0,1) and B1_right (k=2,3).
    # A1 * B1_left (M_{2,2,2})
    # A1 * B1_right (M_{2,2,2})
    
    # Strassen for A1 (j=0,1) and B1_left (j=0,1, k=0,1)
    # A elements: U[0,0], U[0,1], U[1,0], U[1,1]
    # B elements: V[0,0], V[0,1], V[1,0], V[1,1]
    # W elements: W[0,0], W[1,0], W[0,1], W[1,1] (k=0,1; i=0,1)
    
    # I will generate the 50 terms programmatically to avoid manual index errors.
    
    terms = []
    
    # Helper to add Strassen terms for a 2x2x2 block
    # a, b, c, d are U indices
    # e, f, g, h are V indices
    # w00, w01, w10, w11 are W indices
    def add_strassen_block(a, b, c, d, e, f, g, h, w00, w01, w10, w11):
        # m1 = (a+d)(e+h) -> w00 += 1, w11 += 1
        terms.append((a, e, w00, 1, 1, 1))
        terms.append((d, e, w00, 1, 1, 1))
        terms.append((a, h, w00, 1, 1, 1))
        terms.append((d, h, w00, 1, 1, 1))
        terms.append((a, e, w11, 1, 1, 1))
        terms.append((d, e, w11, 1, 1, 1))
        terms.append((a, h, w11, 1, 1, 1))
        terms.append((d, h, w11, 1, 1, 1))
        
        # m2 = (c+d)e -> w10 += 1, w11 -= 1
        terms.append((c, e, w10, 1, 1, 1))
        terms.append((d, e, w10, 1, 1, 1))
        terms.append((c, e, w11, 1, 1, -1))
        terms.append((d, e, w11, 1, 1, -1))
        
        # m3 = a(f-h) -> w01 += 1, w11 += 1
        terms.append((a, f, w01, 1, 1, 1))
        terms.append((a, h, w01, 1, 1, -1))
        terms.append((a, f, w11, 1, 1, 1))
        terms.append((a, h, w11, 1, 1, -1))
        
        # m4 = d(g-e) -> w00 -= 1, w10 += 1
        terms.append((d, g, w00, 1, 1, -1))
        terms.append((d, e, w00, 1, 1, 1))
        terms.append((d, g, w10, 1, 1, 1))
        terms.append((d, e, w10, 1, 1, -1))
        
        # m5 = (a+b)h -> w00 -= 1, w01 += 1
        terms.append((a, h, w00, 1, 1, -1))
        terms.append((b, h, w00, 1, 1, -1))
        terms.append((a, h, w01, 1, 1, 1))
        terms.append((b, h, w01, 1, 1, 1))
        
        # m6 = (c-a)(e+f) -> w11 += 1
        terms.append((c, e, w11, 1, 1, 1))
        terms.append((c, f, w11, 1, 1, 1))
        terms.append((a, e, w11, 1, 1, -1))
        terms.append((a, f, w11, 1, 1, -1))
        
        # m7 = (b-d)(g+h) -> w00 += 1
        terms.append((b, g, w00, 1, 1, 1))
        terms.append((b, h, w00, 1, 1, 1))
        terms.append((d, g, w00, 1, 1, -1))
        terms.append((d, h, w00, 1, 1, -1))

    # This generates 4*7 = 28 terms per block? No, Strassen has 7 multiplications.
    # My add_strassen_block above expands the multiplications into rank-1 terms.
    # This is incorrect. Strassen reduces the number of multiplications, but the tensor rank is the number of multiplications.
    # The rank-1 terms in the tensor decomposition correspond to the multiplications.
    # So I should add 7 terms per Strassen block, not 28.
    
    # Correct Strassen Tensor Terms:
    # 1. (a+d)(e+h) -> U: a,d; V: e,h; W: w00, w11
    # This is not a single rank-1 term. It's a sum of 4 rank-1 terms if expanded.
    # BUT, in tensor decomposition, a "multiplication" is a rank-1 term.
    # Strassen says we can do it with 7 rank-1 terms.
    # The terms are:
    # 1. (a+d) * (e+h) * (w00 + w11) ? No.
    # The tensor is T = sum m_c U_c V_c W_c.
    # For Strassen:
    # m1 = (a+d)(e+h). Contribution to C: m1(w00 + w11).
    # So term 1: U has 1 at a,d. V has 1 at e,h. W has 1 at w00, w11.
    # This is a valid rank-1 term in the tensor space?
    # Yes, U is a vector, V is a vector, W is a vector.
    # U = e_a + e_d. V = e_e + e_h. W = e_w00 + e_w11.
    # Outer product U V^T W^T? No, tensor is sum of rank-1 tensors.
    # A rank-1 tensor is u \otimes v \otimes w.
    # So yes, we can have non-zero entries in multiple positions in U, V, W for a single rank.
    
    # So, for each Strassen block, we add 7 terms.
    
    terms = []
    
    def add_strassen_rank1(a, b, c, d, e, f, g, h, w00, w01, w10, w11):
        # 1. (a+d)(e+h) -> w00, w11
        terms.append(((a,1), (d,1), (e,1), (h,1), (w00,1), (w11,1)))
        # 2. (c+d)e -> w10, -w11
        terms.append(((c,1), (d,1), (e,1), None, (w10,1), (w11,-1)))
        # 3. a(f-h) -> w01, w11
        terms.append(((a,1), None, (f,1), (h,-1), (w01,1), (w11,1)))
        # 4. d(g-e) -> -w00, w10
        terms.append(((d,1), None, (g,1), (e,-1), (w00,-1), (w10,1)))
        # 5. (a+b)h -> -w00, w01
        terms.append(((a,1), (b,1), (h,1), None, (w00,-1), (w01,1)))
        # 6. (c-a)(e+f) -> w11
        terms.append(((c,1), (a,-1), (e,1), (f,1), (w11,1), None))
        # 7. (b-d)(g+h) -> w00
        terms.append(((b,1), (d,-1), (g,1), (h,1), (w00,1), None))

    # Block 1: A1 (j=0,1) * B1_left (j=0,1, k=0,1)
    # A indices: U[0,0]=0, U[0,1]=1, U[1,0]=2, U[1,1]=3
    # B indices: V[0,0]=0, V[0,1]=1, V[1,0]=4, V[1,1]=5
    # W indices: W[0,0]=0, W[1,0]=1, W[0,1]=2, W[1,1]=3
    add_strassen_rank1(0, 1, 2, 3, 0, 1, 4, 5, 0, 2, 1, 3)
    
    # Block 2: A1 (j=0,1) * B1_right (j=0,1, k=2,3)
    # A indices: same
    # B indices: V[0,2]=2, V[0,3]=3, V[1,2]=6, V[1,3]=7
    # W indices: W[0,2]=4, W[1,2]=5, W[0,3]=6, W[1,3]=7
    add_strassen_rank1(0, 1, 2, 3, 2, 3, 6, 7, 4, 6, 5, 7)
    
    # Block 3: A2 (j=2,3) * B2_left (j=2,3, k=0,1)
    # A indices: U[0,2]=4, U[0,3]=5, U[1,2]=6, U[1,3]=7
    # B indices: V[2,0]=8, V[2,1]=9, V[3,0]=12, V[3,1]=13
    # W indices: same as Block 1
    add_strassen_rank1(4, 5, 6, 7, 8, 9, 12, 13, 0, 2, 1, 3)
    
    # Block 4: A2 (j=2,3) * B2_right (j=2,3, k=2,3)
    # A indices: same
    # B indices: V[2,2]=10, V[2,3]=11, V[3,2]=14, V[3,3]=15
    # W indices: same as Block 2
    add_strassen_rank1(4, 5, 6, 7, 10, 11, 14, 15, 4, 6, 5, 7)
    
    # Part 2: M_{2,4,3} (k in 4..6)
    # A1 (j=0,1) * B1 (j=0,1, k=4,5,6) -> M_{2,2,3}
    # A2 (j=2,3) * B2 (j=2,3, k=4,5,6) -> M_{2,2,3}
    
    # M_{2,2,3}: Split B (2x3) into B_left (2x2, k=4,5) and B_right (2x1, k=6).
    # B_left: Strassen (7 terms).
    # B_right: Schoolbook (4 terms).
    
    def add_schoolbook_2x2x1(a, b, c, d, e, f, w0, w1):
        # a e -> w0
        terms.append(((a,1), None, (e,1), None, (w0,1), None))
        # b f -> w0
        terms.append(((b,1), None, (f,1), None, (w0,1), None))
        # c e -> w1
        terms.append(((c,1), None, (e,1), None, (w1,1), None))
        # d f -> w1
        terms.append(((d,1), None, (f,1), None, (w1,1), None))
        
    # Block 5: A1 * B1_left (k=4,5)
    # A: 0,1,2,3
    # B: V[0,4]=16, V[0,5]=17, V[1,4]=20, V[1,5]=21
    # W: W[0,4]=8, W[1,4]=9, W[0,5]=10, W[1,5]=11
    add_strassen_rank1(0, 1, 2, 3, 16, 17, 20, 21, 8, 10, 9, 11)
    
    # Block 6: A1 * B1_right (k=6)
    # A: 0,1,2,3
    # B: V[0,6]=18, V[1,6]=22
    # W: W[0,6]=12, W[1,6]=13
    add_schoolbook_2x2x1(0, 1, 2, 3, 18, 22, 12, 13)
    
    # Block 7: A2 * B2_left (k=4,5)
    # A: 4,5,6,7
    # B: V[2,4]=24, V[2,5]=25, V[3,4]=28, V[3,5]=29
    # W: 8,10,9,11
    add_strassen_rank1(4, 5, 6, 7, 24, 25, 28, 29, 8, 10, 9, 11)
    
    # Block 8: A2 * B2_right (k=6)
    # A: 4,5,6,7
    # B: V[2,6]=26, V[3,6]=30
    # W: 12,13
    add_schoolbook_2x2x1(4, 5, 6, 7, 26, 30, 12, 13)
    
    # Total terms: 4*7 + 2*(7+4) = 28 + 22 = 50.
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(terms):
        u_entries, v_entries, w_entries = term[0:2], term[2:4], term[4:6]
        # Flatten entries
        for entry in u_entries:
            if entry:
                idx, val = entry
                U[idx][c] = val
        for entry in v_entries:
            if entry:
                idx, val = entry
                V[idx][c] = val
        for entry in w_entries:
            if entry:
                idx, val = entry
                W[idx][c] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
