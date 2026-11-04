"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll build the decomposition in parts:
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Part 2: 4x4x2 using block Strassen (rank 28)
    # Part 3: 4x4x1 using schoolbook (rank 16)
    # Total rank: 93
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # === Strassen 2x2x2 coefficients ===
    # A combinations: 7 terms, each 4 coefficients for (0,0),(0,1),(1,0),(1,1)
    a_comb = [
        [1, 0, 0, 1],   # (0,0)+(1,1)
        [0, 0, 1, 1],   # (1,0)+(1,1)
        [1, 0, 0, 0],   # (0,0)
        [0, 0, 0, 1],   # (1,1)
        [1, 1, 0, 0],   # (0,0)+(0,1)
        [-1, 0, 1, 0],  # (1,0)-(0,0)
        [0, 1, 0, -1],  # (0,1)-(1,1)
    ]
    b_comb = [
        [1, 0, 0, 1],   # (0,0)+(1,1)
        [1, 0, 0, 0],   # (0,0)
        [0, 1, 0, -1],  # (0,1)-(1,1)
        [0, 0, 1, -1],  # (1,0)-(0,0)
        [0, 0, 0, 1],   # (1,1)
        [1, 1, 0, 0],   # (0,0)+(0,1)
        [0, 0, 1, 1],   # (1,0)+(1,1)
    ]
    # C combinations: for each term, list of (row, col, coeff)
    c_comb = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1)],
        [(0, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (1, 1, -1)],
    ]
    
    # === Part 1: 4x4x4 (columns 0-3 of B, columns 0-3 of C) ===
    # Block-level combinations for 2x2 blocks
    # A blocks: A11(rows 0-1,cols 0-1), A12(rows 0-1,cols 2-3), A21(rows 2-3,cols 0-1), A22(rows 2-3,cols 2-3)
    # Block index: (block_row, block_col) -> flat: block_row*2+block_col
    # A11->0, A12->1, A21->2, A22->3
    
    # Block-level A combinations (7 terms, 4 block coefficients)
    block_a = [
        [1, 0, 0, 1],   # A11+A22
        [0, 0, 1, 1],   # A21+A22
        [1, 0, 0, 0],   # A11
        [0, 0, 0, 1],   # A22
        [1, 1, 0, 0],   # A11+A12
        [0, 0, 1, -1],  # A21-A11
        [0, 1, 0, -1],  # A12-A22
    ]
    block_b = [
        [1, 0, 0, 1],   # B11+B22
        [1, 0, 0, 0],   # B11
        [0, 1, 0, -1],  # B12-B22
        [0, 0, 1, -1],  # B21-B11
        [0, 0, 0, 1],   # B22
        [1, 1, 0, 0],   # B11+B12
        [0, 0, 1, 1],   # B21+B22
    ]
    # Block-level C combinations
    block_c = [
        [(0, 0, 1), (1, 1, 1)],  # C11+C22
        [(1, 0, 1)],             # C21
        [(0, 1, 1)],             # C12
        [(0, 0, 1), (1, 0, 1)], # C11+C21
        [(0, 0, -1), (0, 1, 1)],# C11-C22... wait
        [(1, 1, 1)],             # C22
        [(0, 0, 1), (1, 1, -1)],# C11-C22
    ]
    
    # For 4x4x4: rank 49
    # U: 16 x 49, V: 16 x 49, W: 16 x 49
    rank1 = 49
    U1 = [[0]*rank1 for _ in range(16)]
    V1 = [[0]*rank1 for _ in range(16)]
    W1 = [[0]*rank1 for _ in range(16)]
    
    for alpha in range(7):
        for beta in range(7):
            r = alpha * 7 + beta
            # U: for each A element (i, j) where i in 0-3, j in 0-3
            for i in range(4):
                for j in range(4):
                    block_row = i // 2
                    block_col = j // 2
                    elem_row = i % 2
                    elem_col = j % 2
                    block_coeff = block_a[alpha][block_row * 2 + block_col]
                    elem_coeff = a_comb[beta][elem_row * 2 + elem_col]
                    U1[i * 4 + j][r] = block_coeff * elem_coeff
            
            # V: for each B element (j, k) where j in 0-3, k in 0-3
            for j in range(4):
                for k in range(4):
                    block_row = j // 2
                    block_col = k // 2
                    elem_row = j % 2
                    elem_col = k % 2
                    block_coeff = block_b[alpha][block_row * 2 + block_col]
                    elem_coeff = b_comb[beta][elem_row * 2 + elem_col]
                    V1[j * 4 + k][r] = block_coeff * elem_coeff
            
            # W: for each C element (k, i) where k in 0-3, i in 0-3
            # W[k*4+i, r]
            for k in range(4):
                for i in range(4):
                    block_row = k // 2
                    block_col = i // 2
                    elem_row = k % 2
                    elem_col = i % 2
                    # Find coefficient from block_c and c_comb
                    block_coeff = 0
                    for (br, bc, coeff) in block_c[alpha]:
                        if br == block_row and bc == block_col:
                            block_coeff = coeff
                            break
                    elem_coeff = 0
                    for (er, ec, coeff) in c_comb[beta]:
                        if er == elem_row and ec == elem_col:
                            elem_coeff = coeff
                            break
                    W1[k * 4 + i][r] = block_coeff * elem_coeff
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # === Part 2: 4x4x2 (columns 4-5 of B, columns 4-5 of C) ===
    # B columns 4-5, C columns 4-5
    # A (4x4), B_sub (4x2), C_sub (4x2)
    # Split into 2x2 blocks:
    # A: A11, A12, A21, A22
    # B_sub: B1 (rows 0-1), B2 (rows 2-3)
    # C_sub: C1 (rows 0-1), C2 (rows 2-3)
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    # Each product is 2x2x2, use Strassen (7 terms)
    # Total: 4 * 7 = 28 terms
    
    rank2 = 28
    U2 = [[0]*rank2 for _ in range(16)]
    V2 = [[0]*rank2 for _ in range(28)]  # m*p = 4*7 = 28
    W2 = [[0]*rank2 for _ in range(28)]  # p*n = 7*4 = 28
    
    # For this part, B columns are 4,5 and C columns are 4,5
    # B element index: j*7 + k where j in 0-3, k in 0-6
    # For this part, k is 4 or 5
    # C element index: k*4 + i where k in 0-6, i in 0-3
    # For this part, k is 4 or 5
    
    # Products: (A_block, B_block, C_block)
    # A11*B1 -> C1: A block 0, B rows 0-1, C rows 0-1
    # A12*B2 -> C1: A block 1, B rows 2-3, C rows 0-1
    # A21*B1 -> C2: A block 2, B rows 0-1, C rows 2-3
    # A22*B2 -> C2: A block 3, B rows 2-3, C rows 2-3
    
    products = [
        (0, 0, 0),  # A11, B rows 0-1, C rows 0-1
        (1, 1, 0),  # A12, B rows 2-3, C rows 0-1
        (2, 0, 1),  # A21, B rows 2-3... wait
    ]
    
    # Let me reconsider. A11 is rows 0-1, cols 0-1 of A.
    # A12 is rows 0-1, cols 2-3 of A.
    # A21 is rows 2-3, cols 0-1 of A.
    # A22 is rows 2-3, cols 2-3 of A.
    
    # B_sub (4x2): rows 0-3, cols 0-1 (which are global cols 4-5)
    # B1 = rows 0-1 of B_sub
    # B2 = rows 2-3 of B_sub
    
    # C_sub (4x2): rows 0-3, cols 0-1 (which are global cols 4-5)
    # C1 = rows 0-1 of C_sub
    # C2 = rows 2-3 of C_sub
    
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    
    # Product 1: A11 * B1 -> contributes to C1
    # A11: A rows 0-1, cols 0-1
    # B1: B rows 0-1, cols 4-5 (global)
    # C1: C rows 0-1, cols 4-5 (global)
    
    # Product 2: A12 * B2 -> contributes to C1
    # A12: A rows 0-1, cols 2-3
    # B2: B rows 2-3, cols 4-5 (global)
    # C1: C rows 0-1, cols 4-5 (global)
    
    # Product 3: A21 * B1 -> contributes to C2
    # A21: A rows 2-3, cols 0-1
    # B1: B rows 0-1, cols 4-5 (global)
    # C2: C rows 2-3, cols 4-5 (global)
    
    # Product 4: A22 * B2 -> contributes to C2
    # A22: A rows 2-3, cols 2-3
    # B2: B rows 2-3, cols 4-5 (global)
    # C2: C rows 2-3, cols 4-5 (global)
    
    products = [
        # (A_row_start, A_col_start, B_row_start, B_col_start, C_row_start, C_col_start)
        (0, 0, 0, 4, 0, 4),  # A11 * B1 -> C1
        (0, 2, 2, 4, 0, 4),  # A12 * B2 -> C1
        (2, 0, 0, 4, 2, 4),  # A21 * B1 -> C2
        (2, 2, 2, 4, 2, 4),  # A22 * B2 -> C2
    ]
    
    for prod_idx, (ar, ac, br, bc, cr, cc) in enumerate(products):
        for beta in range(7):
            r = prod_idx * 7 + beta
            # U: A elements in rows ar..ar+1, cols ac..ac+1
            for i in range(ar, ar+2):
                for j in range(ac, ac+1):
                    elem_row = i - ar
                    elem_col = j - ac
                    U2[i * 4 + j][r] = a_comb[beta][elem_row * 2 + elem_col]
            
            # V: B elements in rows br..br+1, cols bc..bc+1
            for j in range(br, br+2):
                for k in range(bc, bc+2):
                    elem_row = j - br
                    elem_col = k - bc
                    V2[j * 7 + k][r] = b_comb[beta][elem_row * 2 + elem_col]
            
            # W: C elements in rows cr..cr+1, cols cc..cc+1
            # W[k*4+i, r] where k is row of C, i is col of C
            for k in range(cr, cr+2):
                for i in range(cc, cc+1):
                    elem_row = k - cr
                    elem_col = i - cc
                    coeff = 0
                    for (er, ec, c) in c_comb[beta]:
                        if er == elem_row and ec == elem_col:
                            coeff = c
                            break
                    W2[k * 4 + i][r] = coeff
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # === Part 3: 4x4x1 (column 6 of B, column 6 of C) ===
    # A (4x4), B_sub (4x1), C_sub (4x1)
    # Split into 2x2 blocks:
    # A: A11, A12, A21, A22
    # B_sub: B1 (rows 0-1), B2 (rows 2-3)
    # C_sub: C1 (rows 0-1), C2 (rows 2-3)
    # C1 = A11*B1 + A12*B2
    # C2 = A21*B1 + A22*B2
    # Each product is 2x2x1, rank 4
    # Total: 4 * 4 = 16 terms
    
    rank3 = 16
    U3 = [[0]*rank3 for _ in range(16)]
    V3 = [[0]*rank3 for _ in range(28)]
    W3 = [[0]*rank3 for _ in range(28)]
    
    # 2x2x1 decomposition: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # 4 terms:
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    # Term 2: A[1,0]*B[0,0] -> C[1,0]
    # Term 3: A[1,1]*B[1,0] -> C[1,0]
    
    # For 2x2x1, the decomposition is:
    # U[0,0]=1, V[0,0]=1, W[0,0]=1  (C[0,0] += A[0,0]*B[0,0])
    # U[1,0]=1, V[2,0]=1, W[0,0]=1  (C[0,0] += A[0,1]*B[1,0])
    # U[2,0]=1, V[0,0]=1, W[1,0]=1  (C[1,0] += A[1,0]*B[0,0])
    # U[3,0]=1, V[2,0]=1, W[1,0]=1  (C[1,0] += A[1,1]*B[1,0])
    
    # In the 2x2x1 case: n=2, m=2, p=1
    # U[i*2+j, r], V[j*1+k, r], W[k*2+i, r]
    # U: 4x4, V: 2x4, W: 2x4
    
    # products for 4x4x1:
    products3 = [
        # (A_row_start, A_col_start, B_row_start, C_row_start)
        (0, 0, 0, 0),  # A11 * B1 -> C1
        (0, 2, 2, 0),  # A12 * B2 -> C1
        (2, 0, 0, 2),  # A21 * B1 -> C2
        (2, 2, 2, 2),  # A22 * B2 -> C2
    ]
    
    for prod_idx, (ar, ac, br, cr) in enumerate(products3):
        # 4 terms for 2x2x1
        terms = [
            # (A_elem_row, A_elem_col, B_elem_row, C_elem_row)
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1),
        ]
        for term_idx, (aer, aec, ber, cer) in enumerate(terms):
            r = prod_idx * 4 + term_idx
            # U
            i = ar + aer
            j = ac + aec
            U3[i * 4 + j][r] = 1
            # V
            jj = br + ber
            kk = 6  # global col 6
            V3[jj * 7 + kk][r] = 1
            # W
            cc_row = cr + cer
            cc_col = 6  # global col 6
            W3[cc_row * 4 + cc_col][r] = 1
    
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    # Combine all parts
    total_rank = rank1 + rank2 + rank3
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    offset = 0
    for part_idx in range(3):
        r = rank1 if part_idx == 0 else (rank2 if part_idx == 1 else rank3)
        for i in range(16):
            for c in range(r):
                U[i][offset + c] = U_parts[part_idx][i][c]
        for i in range(28):
            for c in range(r):
                V[i][offset + c] = V_parts[part_idx][i][c]
        for i in range(28):
            for c in range(r):
                W[i][offset + c] = W_parts[part_idx][i][c]
        offset += r
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
