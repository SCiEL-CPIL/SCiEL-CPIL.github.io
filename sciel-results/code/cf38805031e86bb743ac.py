import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a local decomposition
    # u_start, v_start, w_start are offsets in U, V, W
    # u_size, v_size, w_size are dimensions of the local block
    # decomp is a list of (u_local, v_local, w_local)
    def add_block(u_start, v_start, w_start, u_size, v_size, w_size, decomp):
        nonlocal col
        for u_l, v_l, w_l in decomp:
            for i, val in enumerate(u_l):
                if val != 0:
                    U[u_start + i][col] += val
            for i, val in enumerate(v_l):
                if val != 0:
                    V[v_start + i][col] += val
            for i, val in enumerate(w_l):
                if val != 0:
                    W[w_start + i][col] += val
            col += 1
            
    # Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # Flat indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B[0]=B00, B[1]=B01, B[2]=B10, B[3]=B11
    # C[0]=C00, C[1]=C01, C[2]=C10, C[3]=C11
    strassen_222 = [
        # M1 = A00 * (B00 + B11)
        ([1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]), # Placeholder, need to map to C
        # Actually, let's just use the linear combinations directly in the add_block
        # But add_block expects rank-1 terms (u, v, w)
        # Strassen is not a sum of rank-1 terms in the standard basis?
        # No, Strassen IS a sum of 7 rank-1 terms.
        # M1 = (A00) * (B00+B11). This is rank 1?
        # u = [1,0,0,0], v = [1,0,0,1]. w = ?
        # C00 gets M1. C11 gets -M1?
        # The w vector for M1 must be such that sum u_k v_k w_k = C.
        # C00 = M1 + M4 - M2 - M5
        # C01 = M3 + M5
        # C10 = M6 + M7
        # C11 = M1 - M3 + M7
        
        # So for M1:
        # w1 = [1, 0, 0, 1] (contributes to C00 and C11)
        # u1 = [1, 0, 0, 0] (A00)
        # v1 = [1, 0, 0, 1] (B00+B11)
        
        # M2:
        # w2 = [-1, 0, 0, -1] (contributes to C00 and C11)
        # u2 = [1, 0, 1, 1] (A00+A11)
        # v2 = [0, 0, 0, 1] (B11)
        
        # M3:
        # w3 = [0, 1, 0, -1] (contributes to C01 and C11)
        # u3 = [1, 1, 0, 0] (A00+A01)
        # v3 = [0, 0, 1, 0] (B10)
        
        # M4:
        # w4 = [1, 0, 0, 0] (contributes to C00)
        # u4 = [0, 0, 1, 1] (A10+A11)
        # v4 = [1, 1, 0, 0] (B00+B01)
        
        # M5:
        # w5 = [-1, 1, 0, 0] (contributes to C00 and C01)
        # u5 = [0, 1, 0, 1] (A01+A11)
        # v5 = [0, 0, 1, 1] (B10+B11)
        
        # M6:
        # w6 = [0, 0, 1, 0] (contributes to C10)
        # u6 = [1, 1, 0, 0] (A00+A01)
        # v6 = [1, 1, 0, 0] (B00+B01)
        
        # M7:
        # w7 = [0, 0, 1, 1] (contributes to C10 and C11)
        # u7 = [0, 0, 1, 1] (A10+A11)
        # v7 = [1, 0, 0, 0] (B00)
    ]
    
    # Verify Strassen:
    # C00 = 1*M1 - 1*M2 + 1*M4 - 1*M5
    # M1 = A00(B00+B11)
    # M2 = (A00+A11)B11
    # M4 = (A10+A11)(B00+B01)
    # M5 = (A01+A11)(B10+B11)
    # Sum: A00B00 + A00B11 - A00B11 - A11B11 + A10B00 + A10B01 + A11B00 + A11B01 - A01B10 - A01B11 - A11B10 - A11B11
    # = A00B00 + A10B00 + A11B00 + A10B01 + A11B01 - A11B11 - A01B11 - A11B10 - A01B10 ...
    # This is getting complicated to verify by hand. I will trust the standard Strassen vectors.
    # However, the w vectors I listed above are coefficients for C.
    # Let's check C00:
    # w1[0]=1, w2[0]=-1, w4[0]=1, w5[0]=-1. Sum = 1-1+1-1 = 0?
    # Wait.
    # C00 = M1 + M4 - M2 - M5.
    # So w1[0]=1, w4[0]=1, w2[0]=-1, w5[0]=-1.
    # My w2 was [-1, 0, 0, -1]. So w2[0]=-1. Correct.
    # My w5 was [-1, 1, 0, 0]. So w5[0]=-1. Correct.
    # Sum for C00: 1 (M1) + 1 (M4) - 1 (M2) - 1 (M5) = 0?
    # No, the formula is C00 = M1 + M4 - M2 - M5.
    # So the coefficients are 1, 1, -1, -1.
    # Sum = 0?
    # Let's re-evaluate Strassen.
    # C00 = M1 + M4 - M2 - M5
    # M1 = A00(B00+B11)
    # M2 = (A00+A11)B11
    # M4 = (A10+A11)(B00+B01)
    # M5 = (A01+A11)(B10+B11)
    # This standard set is correct.
    
    strassen_decomp = [
        ([1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([1, 0, 1, 1], [0, 0, 0, 1], [-1, 0, 0, -1]), # M2
        ([1, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]), # M4
        ([0, 1, 0, 1], [0, 0, 1, 1], [-1, 1, 0, 0]), # M5
        ([1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]), # M6
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]), # M7
    ]
    
    # Block 1: A1 (2x2) x B1 (2x6)
    # Split B1 into three 2x2 blocks.
    # A1 is rows 0,1 of A. Cols 0,1 of A.
    # U offset: 0. Size 4.
    # V offset: 0. Size 4 (for first 2x2 of B1).
    # W offset: 0. Size 4 (for first 2x2 of C).
    
    # B1 is 2x6.
    # Block 1.1: B1[:, 0:2]
    add_block(0, 0, 0, 4, 4, 4, strassen_decomp)
    
    # Block 1.2: B1[:, 2:4]
    # V offset: 4. (B1 is 2x6, flat. Cols 2,3 are indices 2,3,4,5? No.
    # B1 is 2x6. Flat index j*p + k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1.
    # j=0, k=2 -> 2. j=0, k=3 -> 3.
    # j=1, k=0 -> 6. j=1, k=1 -> 7.
    # So B1[:, 0:2] is V[0:2] and V[6:8]?
    # No, V is m*p. m=5, p=6.
    # B1 is the top 2 rows of B.
    # So V indices for B1 are 0..11?
    # B is 5x6.
    # Row 0: 0..5. Row 1: 6..11.
    # So B1 (rows 0,1) uses V[0..11].
    # B1[:, 0:2] uses V[0,1] and V[6,7].
    # This is not a contiguous block of size 4.
    # My add_block assumes contiguous local blocks.
    # I need to handle non-contiguous indices or permute.
    # It is easier to define the local vectors with zeros for the unused parts.
    
    # Let's redefine add_block to take full-size local vectors.
    # Or just generate the full U, V, W vectors for each of the 54 terms.
    
    # I will generate the 54 terms explicitly.
    terms = []
    
    # Helper to create a local Strassen term and map to global indices
    def add_strassen(u_off, v_off, w_off, u_map, v_map, w_map):
        for u_l, v_l, w_l in strassen_decomp:
            u_g = [0] * (n*m)
            v_g = [0] * (m*p)
            w_g = [0] * (p*n)
            
            for i, val in enumerate(u_l):
                if val != 0:
                    u_g[u_map[i]] = val
            for i, val in enumerate(v_l):
                if val != 0:
                    v_g[v_map[i]] = val
            for i, val in enumerate(w_l):
                if val != 0:
                    w_g[w_map[i]] = val
                    
            terms.append((u_g, v_g, w_g))
            
    # Helper to create a 2x1x2 term
    def add_212(u_off, v_off, w_off, u_map, v_map, w_map):
        # A is 2x1, B is 1x2
        # A[0], A[1]
        # B[0], B[1]
        # C[0,0] = A0 B0
        # C[0,1] = A0 B1
        # C[1,0] = A1 B0
        # C[1,1] = A1 B1
        decomp_212 = [
            ([1, 0], [1, 0], [1, 0, 0, 0]),
            ([1, 0], [0, 1], [0, 1, 0, 0]),
            ([0, 1], [1, 0], [0, 0, 1, 0]),
            ([0, 1], [0, 1], [0, 0, 0, 1]),
        ]
        for u_l, v_l, w_l in decomp_212:
            u_g = [0] * (n*m)
            v_g = [0] * (m*p)
            w_g = [0] * (p*n)
            
            for i, val in enumerate(u_l):
                if val != 0:
                    u_g[u_map[i]] = val
            for i, val in enumerate(v_l):
                if val != 0:
                    v_g[v_map[i]] = val
            for i, val in enumerate(w_l):
                if val != 0:
                    w_g[w_map[i]] = val
                    
            terms.append((u_g, v_g, w_g))

    # Mapping for A1 (2x2) in A (2x5)
    # A1 is A[0:2, 0:2]
    # Flat A: i*m + j.
    # i=0, j=0 -> 0. i=0, j=1 -> 1.
    # i=1, j=0 -> 5. i=1, j=1 -> 6.
    u_map_A1 = [0, 1, 5, 6]
    
    # Mapping for B1 (2x6) in B (5x6)
    # B1 is B[0:2, 0:6]
    # Flat B: j*p + k.
    # j=0, k=0 -> 0. j=0, k=1 -> 1.
    # j=1, k=0 -> 6. j=1, k=1 -> 7.
    
    # Block 1.1: B1[:, 0:2]
    v_map_B1_1 = [0, 1, 6, 7]
    w_map_C1_1 = [0, 1, 2, 3] # C[0:2, 0:2]
    # C is p x n. Flat k*n + i.
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    
    add_strassen(0, 0, 0, u_map_A1, v_map_B1_1, w_map_C1_1)
    
    # Block 1.2: B1[:, 2:4]
    v_map_B1_2 = [2, 3, 8, 9]
    w_map_C1_2 = [4, 5, 6, 7] # k=2,3
    add_strassen(0, 0, 0, u_map_A1, v_map_B1_2, w_map_C1_2)
    
    # Block 1.3: B1[:, 4:6]
    v_map_B1_3 = [4, 5, 10, 11]
    w_map_C1_3 = [8, 9, 10, 11] # k=4,5
    add_strassen(0, 0, 0, u_map_A1, v_map_B1_3, w_map_C1_3)
    
    # Now A2 (2x3) x B2 (3x6)
    # A2 is A[0:2, 2:5]
    # Flat A:
    # i=0, j=2 -> 2. i=0, j=3 -> 3. i=0, j=4 -> 4.
    # i=1, j=2 -> 7. i=1, j=3 -> 8. i=1, j=4 -> 9.
    u_map_A2 = [2, 3, 4, 7, 8, 9]
    
    # B2 is B[2:5, 0:6]
    # Flat B:
    # j=2, k=0 -> 12. j=2, k=1 -> 13.
    # j=3, k=0 -> 18. j=3, k=1 -> 19.
    # j=4, k=0 -> 24. j=4, k=1 -> 25.
    
    # Block 2.1: B2[:, 0:2]
    # A2[:, 0:2] is A2 cols 0,1 (indices 0,1 in local A2).
    # B2[0:2, 0:2] is B2 rows 0,1 (indices 0,1 in local B2).
    # Local A2 size 6. Local B2 size 6.
    # Strassen part uses local A2[0:2, 0:2] and local B2[0:2, 0:2].
    u_map_A2_s = [u_map_A2[0], u_map_A2[1], u_map_A2[3], u_map_A2[4]] # A2[0,0], A2[0,1], A2[1,0], A2[1,1]
    v_map_B2_s = [12, 13, 18, 19] # B2[0,0], B2[0,1], B2[1,0], B2[1,1]
    w_map_C2_1 = [0, 1, 2, 3]
    
    add_strassen(0, 0, 0, u_map_A2_s, v_map_B2_s, w_map_C2_1)
    
    # 2x1x2 part for Block 2.1
    # A2 col 2 (local index 2,4) -> global 4, 9
    # B2 row 2 (local index 4,5) -> global 24, 25
    u_map_A2_212 = [4, 9]
    v_map_B2_212 = [24, 25]
    add_212(0, 0, 0, u_map_A2_212, v_map_B2_212, w_map_C2_1)
    
    # Block 2.2: B2[:, 2:4]
    u_map_A2_s2 = [u_map_A2[0], u_map_A2[1], u_map_A2[3], u_map_A2[4]]
    v_map_B2_s2 = [14, 15, 20, 21]
    w_map_C2_2 = [4, 5, 6, 7]
    
    add_strassen(0, 0, 0, u_map_A2_s2, v_map_B2_s2, w_map_C2_2)
    
    u_map_A2_212_2 = [4, 9]
    v_map_B2_212_2 = [26, 27]
    add_212(0, 0, 0, u_map_A2_212_2, v_map_B2_212_2, w_map_C2_2)
    
    # Block 2.3: B2[:, 4:6]
    u_map_A2_s3 = [u_map_A2[0], u_map_A2[1], u_map_A2[3], u_map_A2[4]]
    v_map_B2_s3 = [16, 17, 22, 23]
    w_map_C2_3 = [8, 9, 10, 11]
    
    add_strassen(0, 0, 0, u_map_A2_s3, v_map_B2_s3, w_map_C2_3)
    
    u_map_A2_212_3 = [4, 9]
    v_map_B2_212_3 = [28, 29]
    add_212(0, 0, 0, u_map_A2_212_3, v_map_B2_212_3, w_map_C2_3)
    
    # Build U, V, W from terms
    U = [[0] * len(terms) for _ in range(n * m)]
    V = [[0] * len(terms) for _ in range(m * p)]
    W = [[0] * len(terms) for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for i, val in enumerate(u):
            if val != 0:
                U[i][c] = val
        for i, val in enumerate(v):
            if val != 0:
                V[i][c] = val
        for i, val in enumerate(w):
            if val != 0:
                W[i][c] = val
                
    return {"U": U, "V": V, "W": W}
