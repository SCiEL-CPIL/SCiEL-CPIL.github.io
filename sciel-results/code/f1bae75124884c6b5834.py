import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 72.
    # Part 1: 4x4x4 using Strassen (7 blocks) x Schoolbook (8 terms) = 56 terms.
    # Part 2: 4x4x1 using Schoolbook = 16 terms.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: list of (idx, coeff) for U
        # v_coeffs: list of (idx, coeff) for V
        # w_coeffs: list of (idx, coeff) for W
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, c in u_coeffs:
            U[idx][col] += c
        for idx, c in v_coeffs:
            V[idx][col] += c
        for idx, c in w_coeffs:
            W[idx][col] += c
        col += 1

    # --- Part 1: 4x4x4 ---
    # Strassen blocks
    # A blocks:
    # A11: A[0:2, 0:2] -> indices 0,1,4,5
    # A12: A[0:2, 2:4] -> indices 2,3,6,7
    # A21: A[2:4, 0:2] -> indices 8,9,12,13
    # A22: A[2:4, 2:4] -> indices 10,11,14,15
    
    # B blocks (4x4):
    # B11: B[0:2, 0:2] -> indices 0,1,4,5
    # B12: B[0:2, 2:4] -> indices 2,3,6,7
    # B21: B[2:4, 0:2] -> indices 8,9,12,13
    # B22: B[2:4, 2:4] -> indices 10,11,14,15
    
    # We define the 7 Strassen products M1..M7.
    # Each Mk is a 2x2 matrix product Xk * Yk.
    # Xk and Yk are 2x2 matrices of linear forms.
    
    # We need to define the linear forms for X and Y for each block.
    # Then for each of the 8 schoolbook terms in X*Y, we add a term.
    
    # Let's define the blocks as lists of (A_index, coeff)
    # A_index is i*m + j
    
    def get_A_block(r, c):
        # r, c are block indices 0 or 1
        # returns list of (global_index, coeff)
        res = []
        for i in range(2):
            for j in range(2):
                gi = (r * 2 + i) * 4 + (c * 2 + j)
                res.append((gi, 1))
        return res

    def get_B_block(r, c):
        res = []
        for i in range(2):
            for j in range(2):
                gi = (r * 2 + i) * 4 + (c * 2 + j)
                res.append((gi, 1))
        return res

    # Strassen definitions:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We represent X and Y as 2x2 lists of lists of (coeff, source)
    # source is 'A' or 'B'
    # Actually, we just need the coefficients for the global indices.
    
    # Let's create a function to process one Strassen block
    def process_block(X_blocks, Y_blocks, C_contribs):
        # X_blocks: 2x2 list of lists of (A_idx, coeff)
        # Y_blocks: 2x2 list of lists of (B_idx, coeff)
        # C_contribs: dict mapping (row, col) of M to list of (C_idx, coeff)
        # C_idx is k*n + i for W? No, W index is k*n + i.
        # C is 4x4. C[i, k].
        # M is 2x2. M[0,0] -> C[0,0], M[0,1] -> C[0,1], M[1,0] -> C[1,0], M[1,1] -> C[1,1]
        # But M is a block of C.
        # M1 is top-left block of C? No.
        # C11 is top-left 2x2 of C.
        # So M1[0,0] adds to C[0,0].
        
        # Schoolbook terms for 2x2 multiplication:
        # T1 = X00 * Y00 -> adds to M00
        # T2 = X01 * Y10 -> adds to M00
        # T3 = X00 * Y01 -> adds to M01
        # T4 = X01 * Y11 -> adds to M01
        # T5 = X10 * Y00 -> adds to M10
        # T6 = X11 * Y10 -> adds to M10
        # T7 = X10 * Y01 -> adds to M11
        # T8 = X11 * Y11 -> adds to M11
        
        terms = [
            ((0,0), (0,0), (0,0)), # X00, Y00 -> M00
            ((0,1), (1,0), (0,0)), # X01, Y10 -> M00
            ((0,0), (0,1), (0,1)), # X00, Y01 -> M01
            ((0,1), (1,1), (0,1)), # X01, Y11 -> M01
            ((1,0), (0,0), (1,0)), # X10, Y00 -> M10
            ((1,1), (1,0), (1,0)), # X11, Y10 -> M10
            ((1,0), (0,1), (1,1)), # X10, Y01 -> M11
            ((1,1), (1,1), (1,1)), # X11, Y11 -> M11
        ]
        
        for x_idx, y_idx, m_idx in terms:
            u_coeffs = X_blocks[x_idx[0]][x_idx[1]]
            v_coeffs = Y_blocks[y_idx[0]][y_idx[1]]
            
            # Determine W coefficients
            # m_idx is (r, c) in the 2x2 block M.
            # This block M corresponds to a specific 2x2 sub-block of C.
            # We need to know which sub-block of C this M is part of.
            # This is handled by C_contribs passed in?
            # Actually, the Strassen formula defines how M1..M7 combine to form C blocks.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So for Block 1 (M1):
            # M1[0,0] contributes to C[0,0] (coeff 1) and C[2,2] (coeff 1).
            # M1[0,1] contributes to C[0,1] (coeff 1) and C[2,3] (coeff 1).
            # M1[1,0] contributes to C[1,0] (coeff 1) and C[3,2] (coeff 1).
            # M1[1,1] contributes to C[1,1] (coeff 1) and C[3,3] (coeff 1).
            
            # We will pass a function or dict that maps M[r,c] to W indices.
            
            w_coeffs = []
            for c_idx, c_coeff in C_contribs[m_idx]:
                # c_idx is (i, k) in C.
                # W index is k * n + i.
                w_idx = c_idx[1] * n + c_idx[0]
                w_coeffs.append((w_idx, c_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)

    # Define blocks for A and B
    A11 = get_A_block(0,0)
    A12 = get_A_block(0,1)
    A21 = get_A_block(1,0)
    A22 = get_A_block(1,1)
    
    B11 = get_B_block(0,0)
    B12 = get_B_block(0,1)
    B21 = get_B_block(1,0)
    B22 = get_B_block(1,1)
    
    # Helper to create 2x2 block structure
    def make_2x2(b00, b01, b10, b11):
        return [
            [b00, b01],
            [b10, b11]
        ]

    # M1 = (A11+A22)(B11+B22)
    X1 = make_2x2(A11+A22, A11+A22, A11+A22, A11+A22) # Wait, A11+A22 is element-wise?
    # No, A11 is a block. A11+A22 means the sum of the blocks.
    # So X1 is a 2x2 matrix where every entry is the sum of the corresponding entries of A11 and A22?
    # No.
    # $X_1 = A_{11} + A_{22}$.
    # $A_{11}$ is a $2 \times 2$ matrix. $A_{22}$ is a $2 \times 2$ matrix.
    # So $X_1$ is a $2 \times 2$ matrix.
    # $X_1[0,0] = A_{11}[0,0] + A_{22}[0,0]$.
    
    # My get_A_block returns a list of coefficients for the whole block.
    # I need to handle the 2x2 structure inside the block.
    
    # Let's refine the representation.
    # A_block is a 2x2 list of lists of (global_idx, coeff).
    
    def get_A_block_struct(r, c):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                gi = (r * 2 + i) * 4 + (c * 2 + j)
                res[i][j].append((gi, 1))
        return res

    def get_B_block_struct(r, c):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                gi = (r * 2 + i) * 4 + (c * 2 + j)
                res[i][j].append((gi, 1))
        return res

    A11_s = get_A_block_struct(0,0)
    A12_s = get_A_block_struct(0,1)
    A21_s = get_A_block_struct(1,0)
    A22_s = get_A_block_struct(1,1)
    
    B11_s = get_B_block_struct(0,0)
    B12_s = get_B_block_struct(0,1)
    B21_s = get_B_block_struct(1,0)
    B22_s = get_B_block_struct(1,1)
    
    def add_blocks(B1, B2):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = B1[i][j] + B2[i][j]
        return res

    def sub_blocks(B1, B2):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = B1[i][j] + [(-1, idx) for idx, _ in B2[i][j]]
        return res

    # M1
    X1 = add_blocks(A11_s, A22_s)
    Y1 = add_blocks(B11_s, B22_s)
    # C contributions for M1:
    # C11 (top-left) and C22 (bottom-right)
    # M1[0,0] -> C[0,0], C[2,2]
    # M1[0,1] -> C[0,1], C[2,3]
    # M1[1,0] -> C[1,0], C[3,2]
    # M1[1,1] -> C[1,1], C[3,3]
    C1_contribs = {
        (0,0): [((0,0), 1), ((2,2), 1)],
        (0,1): [((0,1), 1), ((2,3), 1)],
        (1,0): [((1,0), 1), ((3,2), 1)],
        (1,1): [((1,1), 1), ((3,3), 1)]
    }
    process_block(X1, Y1, C1_contribs)

    # M2 = (A21+A22)B11
    X2 = add_blocks(A21_s, A22_s)
    Y2 = B11_s
    # C21 (bottom-left) and C22 (bottom-right) with -1
    # C21 = M2 + M4. M2 contributes +1 to C21.
    # C22 = M1 - M2 + ... M2 contributes -1 to C22.
    # M2[0,0] -> C[2,0] (+1), C[2,2] (-1)
    # M2[0,1] -> C[2,1] (+1), C[2,3] (-1)
    # M2[1,0] -> C[3,0] (+1), C[3,2] (-1)
    # M2[1,1] -> C[3,1] (+1), C[3,3] (-1)
    C2_contribs = {
        (0,0): [((2,0), 1), ((2,2), -1)],
        (0,1): [((2,1), 1), ((2,3), -1)],
        (1,0): [((3,0), 1), ((3,2), -1)],
        (1,1): [((3,1), 1), ((3,3), -1)]
    }
    process_block(X2, Y2, C2_contribs)

    # M3 = A11(B12-B22)
    X3 = A11_s
    Y3 = sub_blocks(B12_s, B22_s)
    # C12 (top-right) and C22 (bottom-right)
    # C12 = M3 + M5. M3 contributes +1 to C12.
    # C22 = ... + M3 ... M3 contributes +1 to C22.
    # M3[0,0] -> C[0,2] (+1), C[2,2] (+1)
    # M3[0,1] -> C[0,3] (+1), C[2,3] (+1)
    # M3[1,0] -> C[1,2] (+1), C[3,2] (+1)
    # M3[1,1] -> C[1,3] (+1), C[3,3] (+1)
    C3_contribs = {
        (0,0): [((0,2), 1), ((2,2), 1)],
        (0,1): [((0,3), 1), ((2,3), 1)],
        (1,0): [((1,2), 1), ((3,2), 1)],
        (1,1): [((1,3), 1), ((3,3), 1)]
    }
    process_block(X3, Y3, C3_contribs)

    # M4 = A22(B21-B11)
    X4 = A22_s
    Y4 = sub_blocks(B21_s, B11_s)
    # C11 (top-left) and C21 (bottom-left)
    # C11 = M1 + M4 ... M4 contributes +1 to C11.
    # C21 = M2 + M4. M4 contributes +1 to C21.
    # M4[0,0] -> C[0,0] (+1), C[2,0] (+1)
    # M4[0,1] -> C[0,1] (+1), C[2,1] (+1)
    # M4[1,0] -> C[1,0] (+1), C[3,0] (+1)
    # M4[1,1] -> C[1,1] (+1), C[3,1] (+1)
    C4_contribs = {
        (0,0): [((0,0), 1), ((2,0), 1)],
        (0,1): [((0,1), 1), ((2,1), 1)],
        (1,0): [((1,0), 1), ((3,0), 1)],
        (1,1): [((1,1), 1), ((3,1), 1)]
    }
    process_block(X4, Y4, C4_contribs)

    # M5 = (A11+A12)B22
    X5 = add_blocks(A11_s, A12_s)
    Y5 = B22_s
    # C11 (top-left) with -1 and C12 (top-right) with +1
    # C11 = ... - M5 ...
    # C12 = M3 + M5.
    # M5[0,0] -> C[0,0] (-1), C[0,2] (+1)
    # M5[0,1] -> C[0,1] (-1), C[0,3] (+1)
    # M5[1,0] -> C[1,0] (-1), C[1,2] (+1)
    # M5[1,1] -> C[1,1] (-1), C[1,3] (+1)
    C5_contribs = {
        (0,0): [((0,0), -1), ((0,2), 1)],
        (0,1): [((0,1), -1), ((0,3), 1)],
        (1,0): [((1,0), -1), ((1,2), 1)],
        (1,1): [((1,1), -1), ((1,3), 1)]
    }
    process_block(X5, Y5, C5_contribs)

    # M6 = (A21-A11)(B11+B12)
    X6 = sub_blocks(A21_s, A11_s)
    Y6 = add_blocks(B11_s, B12_s)
    # C22 (bottom-right) with +1
    # C22 = ... + M6
    # M6[0,0] -> C[2,2] (+1)
    # M6[0,1] -> C[2,3] (+1)
    # M6[1,0] -> C[3,2] (+1)
    # M6[1,1] -> C[3,3] (+1)
    C6_contribs = {
        (0,0): [((2,2), 1)],
        (0,1): [((2,3), 1)],
        (1,0): [((3,2), 1)],
        (1,1): [((3,3), 1)]
    }
    process_block(X6, Y6, C6_contribs)

    # M7 = (A12-A22)(B21+B22)
    X7 = sub_blocks(A12_s, A22_s)
    Y7 = add_blocks(B21_s, B22_s)
    # C11 (top-left) with +1
    # C11 = ... + M7
    # M7[0,0] -> C[0,0] (+1)
    # M7[0,1] -> C[0,1] (+1)
    # M7[1,0] -> C[1,0] (+1)
    # M7[1,1] -> C[1,1] (+1)
    C7_contribs = {
        (0,0): [((0,0), 1)],
        (0,1): [((0,1), 1)],
        (1,0): [((1,0), 1)],
        (1,1): [((1,1), 1)]
    }
    process_block(X7, Y7, C7_contribs)
    
    # --- Part 2: 4x4x1 ---
    # C_i = sum_j A_ij B_j
    # B is 4x1.
    # Terms: A[i,j] * B[j,0] -> C[i,0]
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 1 + 0
            w_idx = 0 * 4 + i
            add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
            
    return {"U": U, "V": V, "W": W}

