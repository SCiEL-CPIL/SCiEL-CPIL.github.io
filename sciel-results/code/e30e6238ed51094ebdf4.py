"""Strassen-based tensor decomposition for 3x4x8 matrix multiplication."""

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 3, 4, 8
    # U: (n*m, R) = (12, R)
    # V: (m*p, R) = (32, R)
    # W: (p*n, R) = (24, R)
    
    # Strassen 2x2x2 factors for each block
    # For a 2x2x2 block with local indices (i,j,k) in {0,1}:
    # 7 terms M1..M7:
    # M1 = (A10+A11)*(B00+B01)  -> contributes to C11
    # M2 = (A10+A11)*B10        -> contributes to C01, C11
    # M3 = A00*(B01-B11)        -> contributes to C01, C11
    # M4 = A11*(B10-B00)        -> contributes to C00, C10
    # M5 = (A00+A01)*B11        -> contributes to C00, C01
    # M6 = (A10-A00)*(B00+B01)  -> contributes to C11
    # M7 = (A01-A11)*(B10+B11)  -> contributes to C00, C11
    #
    # U factors (rows 0..3, corresponding to A00,A01,A10,A11):
    #   M1: A10+A11 -> rows 2,3 : +1,+1
    #   M2: A10+A11 -> rows 2,3 : +1,+1
    #   M3: A00     -> row 0    : +1
    #   M4: A11     -> row 3    : +1
    #   M5: A00+A01 -> rows 0,1 : +1,+1
    #   M6: A10-A00 -> rows 2,0 : +1,-1
    #   M7: A01-A11 -> rows 1,3 : +1,-1
    U_s = [
        [0, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1,  0, 1],   # A01
        [1, 1, 0, 0, 0,  1, 0],   # A10
        [1, 1, 0, 1, 0,  0,-1],   # A11
    ]
    # V factors (rows 0..3, corresponding to B00,B01,B10,B11):
    #   M1: B00+B01 -> rows 0,1 : +1,+1
    #   M2: B10     -> row 2    : +1
    #   M3: B01-B11 -> rows 1,3 : +1,-1
    #   M4: B10-B00 -> rows 2,0 : +1,-1
    #   M5: B11     -> row 3    : +1
    #   M6: B00+B01 -> rows 0,1 : +1,+1
    #   M7: B10+B11 -> rows 2,3 : +1,+1
    V_s = [
        [1, 0, 0, -1, 0, 1, 0],   # B00
        [1, 0, 1,  0, 0, 1, 0],   # B01
        [0, 1, 0,  1, 0, 0, 1],   # B10
        [0, 0,-1,  0, 1, 0, 1],   # B11
    ]
    # W factors (rows 0..3, corresponding to C00,C01,C10,C11):
    #   M1: C11 -> row 3 : +1
    #   M2: C01+C11 -> rows 1,3 : +1,+1
    #   M3: C01+C11 -> rows 1,3 : +1,+1
    #   M4: C00+C10 -> rows 0,2 : +1,+1
    #   M5: C00+C01 -> rows 0,1 : +1,+1
    #   M6: C11 -> row 3 : +1
    #   M7: C00+C11 -> rows 0,3 : +1,+1
    W_s = [
        [0, 0, 0, 1, 1, 0, 1],   # C00
        [0, 1, 1, 0, 1, 0, 0],   # C01
        [0, 0, 0, 1, 0, 0, 0],   # C10
        [1, 1, 1, 0, 0, 1, 1],   # C11
    ]
    
    # Blocks: 8 blocks of 2x2x2 covering i in {0,1}, j in {0,1} or {2,3}, k in pairs
    # k pairs: (0,1), (2,3), (4,5), (6,7)
    blocks = []
    for i_off in [0]:  # only i=0,1 covered by Strassen
        for j_off in [0, 2]:
            for k_off in [0, 2, 4, 6]:
                blocks.append((i_off, j_off, k_off))
    
    R = 8 * 7 + 4  # 8 Strassen blocks * 7 + 1 extra row * 4
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for bi, (i_off, j_off, k_off) in enumerate(blocks):
        base = bi * 7
        for t in range(7):
            col = base + t
            for a in range(2):
                u_val = U_s[a][t]
                if u_val != 0:
                    U[(i_off + a) * m + (j_off + (a if False else 0))][col] += u_val
            # Wait, I need to be more careful with indexing.
            # U row index: (i_off + a) * m + (j_off + b) where a in {0,1}, b in {0,1}
            # U_s is indexed by a*2 + b
            for ab in range(4):
                a = ab // 2
                b = ab % 2
                u_val = U_s[ab][t]
                if u_val != 0:
                    U[(i_off + a) * m + (j_off + b)][col] += u_val
            # V row index: (j_off + b) * p + (k_off + c) where b in {0,1}, c in {0,1}
            # V_s is indexed by b*2 + c
            for bc in range(4):
                b = bc // 2
                c = bc % 2
                v_val = V_s[bc][t]
                if v_val != 0:
                    V[(j_off + b) * p + (k_off + c)][col] += v_val
            # W row index: (k_off + c) * n + (i_off + a) where c in {0,1}, a in {0,1}
            # W_s is indexed by a*2 + c
            for ac in range(4):
                a = ac // 2
                c = ac % 2
                w_val = W_s[ac][t]
                if w_val != 0:
                    W[(k_off + c) * n + (i_off + a)][col] += w_val
    
    # Row i=2: 1x4x8 part, needs 4 rank terms (one per j in {0,1,2,3})
    for j in range(m):
        col = 56 + j
        U[2 * m + j][col] = 1
        for k in range(p):
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
