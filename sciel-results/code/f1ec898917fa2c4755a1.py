import time

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > tol:
                        return False
        return True

    best_result = make_schoolbook()
    best_rank = n * m * p

    # 2x2x2 Strassen terms
    # Returns list of (Ud, Vd, Wd) where Ud is dict (li, lj)->coeff, etc.
    def get_2x2x2_terms():
        combos = [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),
            ({(0,0):1}, {(1,0):1,(0,0):-1}, {(0,1):1}),
            ({(1,1):1}, {(0,1):1,(1,1):-1}, {(0,0):1,(1,0):1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}, {(1,1):1}),
            ({(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}, {(0,0):1}),
        ]
        return combos

    inner_terms = get_2x2x2_terms()

    # Outer Strassen steps for 4x4x4
    # Each step: (X_blocks, Y_blocks, C_blocks)
    # X_blocks: list of (block_row, block_col, coeff)
    # Y_blocks: list of (block_row, block_col, coeff)
    # C_blocks: list of (block_row, block_col, coeff)
    # Block indices are 0 or 1.
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C blocks: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B10-B00)       -> C01 += M3
    # M4 = A11(B01-B11)       -> C00 += M4, C10 -= M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B10) -> C11 += M6
    # M7 = (A01-A11)(B01+B11) -> C00 -= M7, C11 += M7
    
    outer_steps = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3
        ([(0,0,1)], [(1,0,1), (0,0,-1)], [(0,1,1)]),
        # M4
        ([(1,1,1)], [(0,1,1), (1,1,-1)], [(0,0,1), (1,0,-1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], [(0,0,-1), (1,1,1)]),
    ]

    # Construct 4x4x4 part
    U44 = {} # (i, j) -> list of coeffs for each term
    V44 = {} # (j, k) -> list of coeffs
    W44 = {} # (k, i) -> list of coeffs
    
    num_44_terms = 0
    for X_blocks, Y_blocks, C_blocks in outer_steps:
        for U_loc, V_loc, W_loc in inner_terms:
            term_idx = num_44_terms
            num_44_terms += 1
            
            # Accumulate U contributions
            # X is a 2x2 matrix formed by summing A blocks.
            # X[i][j] = sum_{(br, bc, c) in X_blocks} A_{br, bc}[i][j] * c
            # Global A index: row = br*2 + i, col = bc*2 + j
            for (li, lj), c_loc in U_loc.items():
                for (br, bc, c_x) in X_blocks:
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    key = (gi, gj)
                    if key not in U44:
                        U44[key] = [0] * num_44_terms
                    U44[key][term_idx] += c_loc * c_x

            # Accumulate V contributions
            # Y is a 2x2 matrix formed by summing B blocks.
            # Y[j][k] = sum_{(br, bc, c) in Y_blocks} B_{br, bc}[j][k] * c
            # Global B index: row = br*2 + j, col = bc*2 + k
            for (lj, lk), c_loc in V_loc.items():
                for (br, bc, c_y) in Y_blocks:
                    gj = br * 2 + lj
                    gk = bc * 2 + lk
                    key = (gj, gk)
                    if key not in V44:
                        V44[key] = [0] * num_44_terms
                    V44[key][term_idx] += c_loc * c_y

            # Accumulate W contributions
            # W_loc is for C block at (li, lk)? No, W_loc is (lk, li) for C[i][k]?
            # In 2x2x2, W is indexed (k, i).
            # C[i][k] = sum_j A[i][j] B[j][k].
            # W_loc[lk][li] corresponds to C[li][lk].
            # The C block is at (cr, cc).
            # Global C index: row = cr*2 + li, col = cc*2 + lk.
            # W global index: (k, i) -> (cc*2 + lk, cr*2 + li).
            for (lk, li), c_loc in W_loc.items():
                for (cr, cc, c_w) in C_blocks:
                    gk = cc * 2 + lk
                    gi = cr * 2 + li
                    key = (gk, gi)
                    if key not in W44:
                        W44[key] = [0] * num_44_terms
                    W44[key][term_idx] += c_loc * c_w

    # Now we have 49 terms for the 4x4x4 part.
    # We need to combine with 80 schoolbook terms for the 4x4x5 part.
    # Total rank = 49 + 80 = 129.
    
    total_rank = num_44_terms + 80
    
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    # Fill 4x4x4 part
    for (gi, gj), coeffs in U44.items():
        for r in range(num_44_terms):
            U[gi * m + gj][r] = coeffs[r]
            
    for (gj, gk), coeffs in V44.items():
        for r in range(num_44_terms):
            V[gj * p + gk][r] = coeffs[r]
            
    for (gk, gi), coeffs in W44.items():
        for r in range(num_44_terms):
            W[gk * n + gi][r] = coeffs[r]
            
    # Fill 4x4x5 schoolbook part
    # Columns 4..8 of B (which are indices 4..8 in p dimension)
    # i in 0..3, j in 0..3, k in 4..8
    col = num_44_terms
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails
        return best_result

