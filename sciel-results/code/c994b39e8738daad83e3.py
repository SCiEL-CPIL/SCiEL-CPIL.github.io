"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term to the decomposition
    # term: (u_indices, v_indices, w_indices)
    # u_indices: list of (idx, coeff)
    # v_indices: list of (idx, coeff)
    # w_indices: list of (idx, coeff)
    def add_term(u, v, w, col):
        for idx, coeff in u:
            U[idx][col] += coeff
        for idx, coeff in v:
            V[idx][col] += coeff
        for idx, coeff in w:
            W[idx][col] += coeff

    # Strassen for 2x2 by 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+h)
    # M4 = (d)(g-h)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    def get_strassen_2x2_2x2(a_start, b_start, c_start, d_start, e_start, f_start, g_start, h_start):
        # Returns list of 7 terms (u, v, w)
        # u is 2x2 flattened (a, b, c, d)
        # v is 2x2 flattened (e, f, g, h)
        # w is 2x2 flattened (C11, C12, C21, C22)
        terms = []
        # M1
        terms.append(([(a_start, 1), (d_start, 1)], [(e_start, 1), (h_start, 1)], [(c_start, 1), (c_start+1, 0), (c_start+2, 0), (c_start+3, 1)])) # Wait, W indices need to be mapped to output C
        # This is getting complicated to map indices.
        # Let's just generate the full 56 rank decomposition explicitly.
        pass

    # Explicit 56 rank construction
    # We split the problem into 4 sub-problems of 2x2 by 2x4.
    # Each 2x2 by 2x4 is split into two 2x2 by 2x2 (Strassen).
    
    # Sub-problem 1: A1 * B11 (A1 is 2x2, B11 is 2x4)
    # A1 indices: 0,1,2,3 (rows 0,1 of A)
    # B11 indices: 0,1,2,3 (cols 0,1 of B, rows 0,1) -> Wait, B is 4x8.
    # B11 is top-left 2x4 of B. Indices: 0,1,4,5 (rows 0,1; cols 0,1) and 2,3,6,7 (rows 0,1; cols 2,3)?
    # No, B is 4x8.
    # B11: rows 0,1; cols 0,1,2,3.
    # B12: rows 0,1; cols 4,5,6,7.
    # B21: rows 2,3; cols 0,1,2,3.
    # B22: rows 2,3; cols 4,5,6,7.
    
    # A1: rows 0,1; cols 0,1.
    # A2: rows 0,1; cols 2,3.
    
    # C1: cols 0,1,2,3 of C.
    # C2: cols 4,5,6,7 of C.
    
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # We need to decompose 2x2 by 2x4.
    # Split 2x4 into two 2x2.
    # B11 = [B11a, B11b]
    # A1*B11 = A1*B11a + A1*B11b
    
    # So we have 8 Strassen 2x2x2x2 multiplications.
    # Each has rank 7.
    # Total 56.
    
    # Let's define the 8 sub-multiplications.
    # 1. A1 * B11a (C1a)
    # 2. A1 * B11b (C1b)
    # 3. A2 * B21a (C1c)
    # 4. A2 * B21b (C1d)
    # 5. A1 * B12a (C2a)
    # 6. A1 * B12b (C2b)
    # 7. A2 * B22a (C2c)
    # 8. A2 * B22b (C2d)
    
    # Indices:
    # A1: 0,1,2,3
    # A2: 4,5,6,7
    # B11a: 0,1,4,5
    # B11b: 2,3,6,7
    # B21a: 8,9,12,13
    # B21b: 10,11,14,15
    # B12a: 16,17,20,21
    # B12b: 18,19,22,23
    # B22a: 24,25,28,29
    # B22b: 26,27,30,31
    
    # C1a: 0,1,4,5
    # C1b: 2,3,6,7
    # C1c: 8,9,12,13
    # C1d: 10,11,14,15
    # C2a: 16,17,20,21
    # C2b: 18,19,22,23
    # C2c: 24,25,28,29
    # C2d: 26,27,30,31
    
    # W indices (p*n = 16):
    # C1a: 0,1,4,5
    # C1b: 2,3,6,7
    # C1c: 8,9,12,13
    # C1d: 10,11,14,15
    # C2a: 16,17,20,21 -> Wait, W is 16x56.
    # W indices are 0..15.
    # C is 2x8.
    # C1a is C[0:2, 0:2] -> W[0:4]
    # C1b is C[0:2, 2:4] -> W[4:8]
    # C1c is C[0:2, 0:2] -> W[0:4] (Add to C1a)
    # C1d is C[0:2, 2:4] -> W[4:8] (Add to C1b)
    # C2a is C[0:2, 4:6] -> W[8:12]
    # C2b is C[0:2, 6:8] -> W[12:16]
    # C2c is C[0:2, 4:6] -> W[8:12] (Add to C2a)
    # C2d is C[0:2, 6:8] -> W[12:16] (Add to C2b)
    
    # We will add terms for each of the 8 Strassen decompositions.
    
    def add_strassen(a_idx, b_idx, w_idx, col_offset):
        # a_idx: [a, b, c, d]
        # b_idx: [e, f, g, h]
        # w_idx: [C11, C12, C21, C22]
        a, b, c, d = a_idx
        e, f, g, h = b_idx
        C11, C12, C21, C22 = w_idx
        
        # M1 = (a+d)(e+h)
        add_term([(a,1), (d,1)], [(e,1), (h,1)], [(C11,1), (C22,1)], col_offset)
        # M2 = (c+d)(e)
        add_term([(c,1), (d,1)], [(e,1)], [(C21,1)], col_offset+1)
        # M3 = (a-b)(f+h)
        add_term([(a,1), (b,-1)], [(f,1), (h,1)], [(C12,1)], col_offset+2)
        # M4 = d(g-h)
        add_term([(d,1)], [(g,1), (h,-1)], [(C11,1), (C21,1)], col_offset+3)
        # M5 = (a+b)h
        add_term([(a,1), (b,1)], [(h,1)], [(C11,-1), (C12,1)], col_offset+4)
        # M6 = (c-a)(e+f)
        add_term([(c,1), (a,-1)], [(e,1), (f,1)], [(C22,1)], col_offset+5)
        # M7 = (b-d)(g+h)
        add_term([(b,1), (d,-1)], [(g,1), (h,1)], [(C11,1), (C22,1)], col_offset+6)
        
    # 1. A1 * B11a
    add_strassen([0,1,2,3], [0,1,4,5], [0,1,4,5], 0)
    # 2. A1 * B11b
    add_strassen([0,1,2,3], [2,3,6,7], [2,3,6,7], 7)
    # 3. A2 * B21a
    add_strassen([4,5,6,7], [8,9,12,13], [0,1,4,5], 14)
    # 4. A2 * B21b
    add_strassen([4,5,6,7], [10,11,14,15], [2,3,6,7], 21)
    # 5. A1 * B12a
    add_strassen([0,1,2,3], [16,17,20,21], [8,9,12,13], 28)
    # 6. A1 * B12b
    add_strassen([0,1,2,3], [18,19,22,23], [10,11,14,15], 35)
    # 7. A2 * B22a
    add_strassen([4,5,6,7], [24,25,28,29], [8,9,12,13], 42)
    # 8. A2 * B22b
    add_strassen([4,5,6,7], [26,27,30,31], [10,11,14,15], 49)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
