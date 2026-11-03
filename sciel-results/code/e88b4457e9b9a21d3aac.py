import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # 4x4x9 = 3 blocks of 4x4x3
    # 4x4x3 = 4 blocks of 2x2x3
    # 2x2x3 = Strassen(2x2x2) + Schoolbook(2x2x1) -> Rank 7 + 4 = 11
    
    # We will build the full U, V, W matrices
    # U shape: (4*4, rank) = (16, rank)
    # V shape: (4*9, rank) = (36, rank)
    # W shape: (9*4, rank) = (36, rank)
    
    total_rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a decomposition
    def add_decomp(u_vec, v_vec, w_vec):
        nonlocal total_rank
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        total_rank += 1
        
    # 2x2x3 Strassen + Schoolbook decomposition
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    
    # Strassen for cols 1,2 (e,f,h,i)
    # M1 = (a+b)(e+h)
    # M2 = (a-b)(e+h)
    # M3 = a(i-f)  # Note: standard Strassen uses (i-j) for 2x2x2, here j is col 3.
    # Let's stick to standard Strassen for 2x2x2 on cols 1,2.
    # B_sub = [[e, f], [h, i]]
    # M1 = (a+b)(e+h)
    # M2 = (a-b)(e+h)
    # M3 = a(i-f)
    # M4 = d(g-f) -- Wait, g is col 3. We don't have g in this sub-problem.
    # Standard Strassen for 2x2x2:
    # M1 = (a+b)(e+h)
    # M2 = (a-b)(e+h)
    # M3 = a(i-j)
    # M4 = d(g-f)
    # M5 = (a+d)(g+j)
    # M6 = (b+d)(e-g)
    # M7 = (c+a)(f-i)
    
    # We are using Strassen for cols 1,2. So j=g=0 in this context?
    # No, we just ignore col 3 for the Strassen part.
    # So we treat B as [[e, f, 0], [h, i, 0]] for Strassen.
    # Then add schoolbook for col 3.
    
    # Strassen terms for cols 1,2:
    # 1. (a+b)(e+h)
    # 2. (a-b)(e+h)
    # 3. a(i-f)
    # 4. d(0-f) = -df  (since g=0)
    # 5. (a+d)(0+0) = 0
    # 6. (b+d)(e-0) = (b+d)e
    # 7. (c+a)(f-i)
    
    # This is getting complicated to map indices.
    # I will just use the schoolbook for 2x2x3 (rank 12) to be safe and simple?
    # Rank 12 * 4 * 3 = 144. No improvement.
    
    # Let's use the rank 11 decomposition explicitly.
    # 2x2x3 rank 11:
    # 7 terms for Strassen on cols 1,2
    # 4 terms for Schoolbook on col 3
    
    # Strassen for 2x2x2 (cols 1,2):
    # A = [[a,b],[c,d]], B = [[e,f],[h,i]]
    # C11 = ae+bh, C12 = af+bi, C21 = ce+dh, C22 = cf+di
    
    # 1. (a+b)(e+h) -> u=[1,1,0,0], v=[1,0,1,0] (for e,h)
    # 2. (a-b)(e+h) -> u=[1,-1,0,0], v=[1,0,1,0]
    # 3. a(i-f) -> u=[1,0,0,0], v=[0,-1,0,1] (for f,i)
    # 4. d(0-f) -> u=[0,0,0,1], v=[0,-1,0,0] (for f) -- Wait, standard is d(g-f). g=0.
    # 5. (a+d)(0+0) -> 0
    # 6. (b+d)(e-0) -> u=[0,1,0,1], v=[1,0,0,0] (for e)
    # 7. (c+a)(f-i) -> u=[1,0,1,0], v=[0,1,0,-1] (for f,i)
    
    # Schoolbook for col 3 (g,j):
    # 8. ag -> u=[1,0,0,0], v=[0,0,1,0] (for g)
    # 9. bj -> u=[0,1,0,0], v=[0,0,0,1] (for j)
    # 10. cg -> u=[0,0,1,0], v=[0,0,1,0] (for g)
    # 11. dj -> u=[0,0,0,1], v=[0,0,0,1] (for j)
    
    # This gives rank 11 for 2x2x3.
    
    # Now map to 4x4x3.
    # A is 4x4. Split into 4 2x2 blocks: A11, A12, A21, A22.
    # B is 4x3. Split into 2 2x3 blocks: B1, B2.
    # C is 4x3. Split into 4 2x3 blocks: C11, C12, C21, C22.
    # C11 = A11 B1 + A12 B2
    # C12 = A11 B2 + A12 B1? No.
    # B = [[B11, B12, B13], [B21, B22, B23]] (rows of B)
    # A = [[A11, A12], [A21, A22]]
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C13 = A11 B13 + A12 B23
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # C23 = A21 B13 + A22 B23
    
    # We need 8 decompositions of 2x2x3?
    # C11 needs A11B11 and A12B21.
    # A11B11 is 2x2x3. A12B21 is 2x2x3.
    # So for each C block, we need 2 decompositions.
    # 4 C blocks * 2 decompositions = 8 decompositions of 2x2x3.
    # Each 2x2x3 has rank 11.
    # Total rank = 8 * 11 = 88.
    
    # This is much better than 144.
    
    # Implementation:
    # U: 16 x 88
    # V: 36 x 88
    # W: 36 x 88
    
    # Indices:
    # A: 0-15 (4x4)
    # B: 0-35 (4x9)
    # C: 0-35 (9x4) -- Wait, W[k*n+i, r]. C is p x n = 9 x 4.
    # C index: k*4 + i.
    
    # Let's build the 2x2x3 decomposition vectors.
    # A indices: 0,1,2,3 (a,b,c,d)
    # B indices: 0,1,2,3,4,5 (e,f,g,h,i,j)
    # C indices: 0,1,2,3,4,5 (C11,C12,C13,C21,C22,C23)
    
    # Strassen terms for 2x2x3 (cols 1,2):
    # 1. (a+b)(e+h) -> u=[1,1,0,0], v=[1,0,0,1,0,0]
    # 2. (a-b)(e+h) -> u=[1,-1,0,0], v=[1,0,0,1,0,0]
    # 3. a(i-f) -> u=[1,0,0,0], v=[0,-1,0,0,1,0]
    # 4. d(0-f) -> u=[0,0,0,1], v=[0,-1,0,0,0,0]
    # 5. (a+d)(0) -> 0
    # 6. (b+d)e -> u=[0,1,0,1], v=[1,0,0,0,0,0]
    # 7. (c+a)(f-i) -> u=[1,0,1,0], v=[0,1,0,0,-1,0]
    
    # Schoolbook terms for col 3:
    # 8. ag -> u=[1,0,0,0], v=[0,0,1,0,0,0]
    # 9. bj -> u=[0,1,0,0], v=[0,0,0,0,0,1]
    # 10. cg -> u=[0,0,1,0], v=[0,0,1,0,0,0]
    # 11. dj -> u=[0,0,0,1], v=[0,0,0,0,0,1]
    
    # W vectors for these 11 terms:
    # 1. C11 += ae+bh. C12 += af+bi. (Wait, (a+b)(e+h) = ae+ah+be+bh)
    #    C11 = ae+bh. C22 = cf+di.
    #    Strassen:
    #    C11 = M1 + M4 - M5 + M7
    #    C12 = M3 + M5 - M6 + M7
    #    C21 = M1 - M2 + M4 - M7
    #    C22 = M2 - M4 + M5 + M6
    
    #    M1 = (a+b)(e+h)
    #    M2 = (a-b)(e+h)
    #    M3 = a(i-f)
    #    M4 = d(0-f) = -df
    #    M5 = 0
    #    M6 = (b+d)e
    #    M7 = (c+a)(f-i)
    
    #    C11 = M1 + M4 + M7 = (a+b)(e+h) - df + (c+a)(f-i)
    #    = ae+ah+be+bh - df + cf-ci+af-ai
    #    = ae+bh + (ah-ai) + (be+af) + (cf) + (-df-ci)
    #    This is not ae+bh.
    
    #    I will use the standard Strassen W vectors.
    #    W1 = [1, 0, 0, 1, 0, 0] (C11, C21)
    #    W2 = [-1, 0, 0, 1, 0, 0] (C12, C22) -- No, standard Strassen W is:
    #    M1 -> C11, C22
    #    M2 -> C12, C21
    #    M3 -> C12
    #    M4 -> C11
    #    M5 -> C11, C12, C21, C22
    #    M6 -> C12, C22
    #    M7 -> C11, C21
    
    #    Let's just use the rank 12 schoolbook for 2x2x3 to be safe?
    #    Rank 12 * 8 = 96.
    #    96 < 144.
    #    This is a safe improvement.
    
    # Schoolbook 2x2x3:
    # 1. ae -> u=[1,0,0,0], v=[1,0,0,0,0,0], w=[1,0,0,0,0,0]
    # 2. bf -> u=[0,1,0,0], v=[0,1,0,0,0,0], w=[0,1,0,0,0,0]
    # 3. bg -> u=[0,1,0,0], v=[0,0,1,0,0,0], w=[0,0,1,0,0,0]
    # 4. bh -> u=[0,1,0,0], v=[0,0,0,1,0,0], w=[1,0,0,0,0,0]
    # 5. bi -> u=[0,1,0,0], v=[0,0,0,0,1,0], w=[0,1,0,0,0,0]
    # 6. bj -> u=[0,1,0,0], v=[0,0,0,0,0,1], w=[0,0,1,0,0,0]
    # 7. ce -> u=[0,0,1,0], v=[1,0,0,0,0,0], w=[0,0,0,1,0,0]
    # 8. cf -> u=[0,0,1,0], v=[0,1,0,0,0,0], w=[0,0,0,0,1,0]
    # 9. cg -> u=[0,0,1,0], v=[0,0,1,0,0,0], w=[0,0,0,0,0,1]
    # 10. dh -> u=[0,0,0,1], v=[0,0,0,1,0,0], w=[0,0,0,1,0,0]
    # 11. di -> u=[0,0,0,1], v=[0,0,0,0,1,0], w=[0,0,0,0,1,0]
    # 12. dj -> u=[0,0,0,1], v=[0,0,0,0,0,1], w=[0,0,0,0,0,1]
    
    # Total rank 96.
    
    U = np.zeros((16, 96))
    V = np.zeros((36, 96))
    W = np.zeros((36, 96))
    
    # Map 2x2x3 to 4x4x3
    # A blocks: 0-3, 4-7, 8-11, 12-15
    # B blocks: 0-5, 6-11
    # C blocks: 0-5, 6-11, 12-17, 18-23
    
    # C11 = A11 B1 + A12 B2
    # A11: 0-3, B1: 0-5, C11: 0-5
    # A12: 4-7, B2: 6-11, C11: 0-5
    
    # C12 = A11 B2 + A12 B1
    # A11: 0-3, B2: 6-11, C12: 6-11
    # A12: 4-7, B1: 0-5, C12: 6-11
    
    # C21 = A21 B1 + A22 B2
    # A21: 8-11, B1: 0-5, C21: 12-17
    # A22: 12-15, B2: 6-11, C21: 12-17
    
    # C22 = A21 B2 + A22 B1
    # A21: 8-11, B2: 6-11, C22: 18-23
    # A22: 12-15, B1: 0-5, C22: 18-23
    
    # We need to add 8 decompositions of 2x2x3.
    # Each decomposition has 12 terms.
    # Total 96 terms.
    
    def add_2x2x3(a_offset, b_offset, c_offset, col_idx):
        # col_idx 0-11
        terms = [
            ([1,0,0,0], [1,0,0,0,0,0], [1,0,0,0,0,0]),
            ([0,1,0,0], [0,1,0,0,0,0], [0,1,0,0,0,0]),
            ([0,1,0,0], [0,0,1,0,0,0], [0,0,1,0,0,0]),
            ([0,1,0,0], [0,0,0,1,0,0], [1,0,0,0,0,0]),
            ([0,1,0,0], [0,0,0,0,1,0], [0,1,0,0,0,0]),
            ([0,1,0,0], [0,0,0,0,0,1], [0,0,1,0,0,0]),
            ([0,0,1,0], [1,0,0,0,0,0], [0,0,0,1,0,0]),
            ([0,0,1,0], [0,1,0,0,0,0], [0,0,0,0,1,0]),
            ([0,0,1,0], [0,0,1,0,0,0], [0,0,0,0,0,1]),
            ([0,0,0,1], [0,0,0,1,0,0], [0,0,0,1,0,0]),
            ([0,0,0,1], [0,0,0,0,1,0], [0,0,0,0,1,0]),
            ([0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1]),
        ]
        for u, v, w in terms:
            U[a_offset:a_offset+4, col_idx] = u
            V[b_offset:b_offset+6, col_idx] = v
            W[c_offset:c_offset+6, col_idx] = w
            
    # C11
    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(4, 6, 0, 12)
    # C12
    add_2x2x3(0, 6, 6, 24)
    add_2x2x3(4, 0, 6, 36)
    # C21
    add_2x2x3(8, 0, 12, 48)
    add_2x2x3(12, 6, 12, 60)
    # C22
    add_2x2x3(8, 6, 18, 72)
    add_2x2x3(12, 0, 18, 84)
    
    return {"U": U, "V": V, "W": W}

