"""Program: rank-reduced exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy:
- Split the 4x6 output into C_left (4x4) = A * B_left and C_right (4x2) = A * B_right.
- C_left: tensor product of two 2x2 Strassen decompositions -> rank 7*7 = 49.
- C_right: schoolbook 4x4 * 4x2 -> rank 32.
- Total rank 81 (vs schoolbook 96).
W uses the trace convention: W index = k*n + i  (output row i, output col k).
U index = i*m + j (A row i, A col j), V index = j*p + k (B row j, B col k).
"""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    U_rows = n * m      # 16
    V_rows = m * p      # 24
    W_rows = p * n      # 24

    # 2x2 Strassen decomposition: rank 7.
    # For X (2x2) * Y (2x2) = Z (2x2), with X=[[a,b],[c,d]], Y=[[e,f],[g,h]], Z=[[q11,q12],[q21,q22]]
    # Strassen:
    # m1 = (a+d)*(e+h)
    # m2 = (c+d)*e
    # m3 = a*(f-h)
    # m4 = d*(g-e)
    # m5 = (a+b)*h
    # m6 = (c-a)*(e+f)
    # m7 = (b-d)*(g+h)
    # q11 = m1 + m4 - m5 + m7
    # q12 = m3 + m5
    # q21 = m2 + m4
    # q22 = m1 - m2 + m3 + m6
    #
    # We encode each Strassen term as (U_factor, V_factor, W_contrib) where
    #   U_factor is a 2x2 matrix M_U such that the local A-block entries are linear combos.
    # Actually simpler: each term t gives for each local (i,j) of A a coefficient u_t[i,j],
    # for each local (j,k) of B a coefficient v_t[j,k], and for each local (i,k) of C a coefficient w_t[i,k].
    # Then the global contribution: U[i,j] gets u_t[i,j], V[j,k] gets v_t[j,k], W[k,i] gets w_t[i,k].

    # Strassen term definitions for 2x2:
    # Each term is (u, v, w) where u,v are 2x2 matrices (coefficients on A and B blocks) and w is 2x2 (coeff on C block).
    # m1 = (a+d)*(e+h): u = [[1,0],[0,1]], v = [[1,1],[1,1]], contributes to all q.
    #   q11 += m1, q12 += 0, q21 += 0, q22 += m1
    # m2 = (c+d)*e: u = [[0,0],[1,1]], v = [[1,0],[0,0]], q21 += m2, q22 -= m2
    # m3 = a*(f-h): u = [[1,0],[0,0]], v = [[0,1],[0,-1]], q12 += m3, q22 += m3
    # m4 = d*(g-e): u = [[0,0],[0,1]], v = [[-1,0],[1,0]], q11 += m4, q21 += m4
    # m5 = (a+b)*h: u = [[1,1],[0,0]], v = [[0,0],[0,1]], q11 -= m5, q12 += m5
    # m6 = (c-a)*(e+f): u = [[-1,0],[1,0]], v = [[1,1],[0,0]], q22 += m6
    # m7 = (b-d)*(g+h): u = [[0,1],[0,-1]], v = [[1,1],[1,1]], q11 += m7

    # Represent 2x2 as flat list of 4: [00,01,10,11]
    def m2(r00, r01, r10, r11):
        return [r00, r01, r10, r11]

    ZERO = m2(0,0,0,0)
    # term: (u, v, w)
    strassen_terms = [
        # m1
        (m2(1,0,0,1), m2(1,1,1,1), m2(1,0,0,1)),
        # m2
        (m2(0,0,1,1), m2(1,0,0,0), m2(0,0,1,-1)),
        # m3
        (m2(1,0,0,0), m2(0,1,0,-1), m2(0,1,0,1)),
        # m4
        (m2(0,0,0,1), m2(-1,0,1,0), m2(1,0,1,0)),
        # m5
        (m2(1,1,0,0), m2(0,0,0,1), m2(-1,1,0,0)),
        # m6
        (m2(-1,0,1,0), m2(1,1,0,0), m2(0,0,0,1)),
        # m7
        (m2(0,1,0,-1), m2(1,1,1,1), m2(1,0,0,0)),
    ]

    rank = 0
    U = [[0] * rank for _ in range(U_rows)] if rank else None
    # We'll build as lists of columns first, then transpose.
    cols_U = []
    cols_V = []
    cols_W = []

    def add_column(u16, v24, w24):
        cols_U.append(u16)
        cols_V.append(v24)
        cols_W.append(w24)

    # --- C_left: 4x4 * 4x4 via tensor product of two 2x2 Strassen ---
    # A 4x4 split into 2x2 blocks:
    #   A = [[A00, A01],[A10, A11]] each 2x2.
    #   A00: global rows 0,1 cols 0,1 -> U indices (i*4+j) for i in 0,1 j in 0,1
    #   A01: global rows 0,1 cols 2,3 -> U indices i in 0,1 j in 2,3
    #   A10: global rows 2,3 cols 0,1 -> U indices i in 2,3 j in 0,1
    #   A11: global rows 2,3 cols 2,3 -> U indices i in 2,3 j in 2,3
    # B_left 4x4 similarly split.
    # C_left 4x4 split into 2x2 blocks C00, C01, C10, C11.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Tensor product: for each pair (t1, t2) in strassen x strassen (49 terms),
    #   u1 acts on the (row-block, col-block) of A, v1 on B, w1 on C.
    #   We interpret: the first Strassen handles the 2x2 block-multiplication at the
    #   "outer" level (block rows x block cols -> block result), the second handles
    #   the inner 2x2 within each block.
    #
    # Concretely, term (t1, t2): t1 = (u1, v1, w1) for the 2x2 block structure,
    # t2 = (u2, v2, w2) for the inner 2x2.
    # Global U coefficient for A[br*2+ri, bc*2+cj] = u1[br, bc] * u2[ri, cj]
    # Global V coefficient for B[br*2+ri, bc*2+cj] = v1[br, bc] * v2[ri, cj]
    # Global W coefficient for C[br*2+ri, bc*2+cj] = w1[br, bc] * w2[ri, cj]
    # and W is indexed by (k*n + i) where i is C row, k is C col.
    #   C row = br*2+ri, C col = bc*2+cj -> W index = (bc*2+cj)*4 + (br*2+ri)

    for t1 in strassen_terms:
        u1, v1, w1 = t1
        for t2 in strassen_terms:
            u2, v2, w2 = t2
            u16 = [0]*16
            v24 = [0]*24
            w24 = [0]*24
            # u1 indices: [br*2+bc] for br in 0,1 bc in 0,1 (2x2 flat: 00,01,10,11)
            # u2 indices: [ri*2+cj] for ri in 0,1 cj in 0,1
            for br in range(2):
                for bc in range(2):
                    a_coeff = u1[br*2+bc]
                    if a_coeff == 0:
                        continue
                    for ri in range(2):
                        for cj in range(2):
                            b_coeff = u2[ri*2+cj]
                            if b_coeff == 0:
                                continue
                            val = a_coeff * b_coeff
                            A_row = br*2 + ri
                            A_col = bc*2 + cj
                            u16[A_row*4 + A_col] += val
                            # V for B_left: B[br*2+ri, bc*2+cj] with B col within left (0..3)
                            v_coeff = v1[br*2+bc]
                            w_coeff = w1[br*2+bc]
                            if v_coeff == 0 or w_coeff == 0:
                                pass
            # Recompute V and W similarly (independent loops)
            for br in range(2):
                for bc in range(2):
                    b_coeff_outer = v1[br*2+bc]
                    if b_coeff_outer == 0:
                        continue
                    for ri in range(2):
                        for cj in range(2):
                            b_coeff_inner = v2[ri*2+cj]
                            if b_coeff_inner == 0:
                                continue
                            val = b_coeff_outer * b_coeff_inner
                            B_row = br*2 + ri
                            B_col = bc*2 + cj   # within B_left (0..3)
                            v24[B_row*6 + B_col] += val
            for br in range(2):
                for bc in range(2):
                    w_coeff_outer = w1[br*2+bc]
                    if w_coeff_outer == 0:
                        continue
                    for ri in range(2):
                        for cj in range(2):
                            w_coeff_inner = w2[ri*2+cj]
                            if w_coeff_inner == 0:
                                continue
                            val = w_coeff_outer * w_coeff_inner
                            C_row = br*2 + ri
                            C_col = bc*2 + cj   # within C_left (0..3)
                            w24[C_col*4 + C_row] += val
            add_column(u16, v24, w24)

    # --- C_right: 4x4 * 4x2 via schoolbook (rank 32) ---
    # C_right[i, k] = sum_j A[i,j] B_right[j, k], B_right cols are global cols 4,5.
    # Schoolbook: one term per (i, j, k) -> 4*4*2 = 32 terms.
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u16 = [0]*16
                v24 = [0]*24
                w24 = [0]*24
                u16[i*4 + j] = 1
                v24[j*6 + (4 + k)] = 1
                w24[(4 + k)*4 + i] = 1
                add_column(u16, v24, w24)

    rank = len(cols_U)
    U = [[0]*rank for _ in range(U_rows)]
    V = [[0]*rank for _ in range(V_rows)]
    W = [[0]*rank for _ in range(W_rows)]
    for r in range(rank):
        for i in range(U_rows):
            if cols_U[r][i] != 0:
                U[i][r] = cols_U[r][i]
        for i in range(V_rows):
            if cols_V[r][i] != 0:
                V[i][r] = cols_V[r][i]
        for i in range(W_rows):
            if cols_W[r][i] != 0:
                W[i][r] = cols_W[r][i]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
