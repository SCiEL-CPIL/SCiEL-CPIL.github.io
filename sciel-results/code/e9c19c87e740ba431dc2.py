"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 4
    
    # Block indices for 4x4 matrices partitioned into 2x2 blocks of 2x2
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # A row index i: block_row = i//2, local_row = i%2
    # A col index j: block_col = j//2, local_col = j%2
    
    # We'll build U, V, W as lists of lists
    U = []  # each element is a length-16 vector
    V = []
    W = []
    
    # 2x2 Strassen: define the 7 multiplications
    # Each multiplication is defined by:
    #   left_coeffs: dict mapping (local_i, local_j) -> coeff for A' entries
    #   right_coeffs: dict mapping (local_j, local_k) -> coeff for B' entries
    #   output_contrib: dict mapping (output_i, output_j) -> coeff for C' entries
    # where local indices are 0,1
    
    # Strassen 2x2 multiplication:
    # m1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    # m2 = (a10+a11)(b00)     -> c10 += 1
    # m3 = (a00)(b01-b11)     -> c01 += 1
    # m4 = (a11)(b10-b00)     -> c10 += 1, c00 += 1
    # m5 = (a00+a01)(b11)     -> c01 += 1, c00 -= 1
    # m6 = (a10-a00)(b00+b01) -> c11 += 1
    # m7 = (a11-a10)(b10+b11) -> c00 += 1
    
    strassen_2x2 = [
        # (left_coeffs, right_coeffs, output_coeffs)
        # m1: (a00+a11)(b00+b11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # m2: (a10+a11)(b00)
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # m3: (a00)(b01-b11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # m4: (a11)(b10-b00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
        # m5: (a00+a01)(b11)
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (0,0):-1}),
        # m6: (a10-a00)(b00+b01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # m7: (a11-a10)(b10+b11)
        ({(1,1):1, (1,0):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Outer Strassen: 7 multiplications of 2x2 blocks
    # Each outer multiplication M_s involves linear combinations of 4 blocks each for A and B
    # A blocks: A00, A01, A10, A11 indexed by (block_i, block_j)
    # B blocks: B00, B01, B10, B11 indexed by (block_j, block_k)
    
    # For each outer multiplication, we specify:
    #   A_block_coeffs: dict mapping (bi, bj) -> coeff
    #   B_block_coeffs: dict mapping (bj, bk) -> coeff
    #   C_block_coeffs: dict mapping (bi, bk) -> coeff (how M_s contributes to C blocks)
    
    outer_strassen = [
        # M1 = (A00+A11)(B00+B11)
        (
            {(0,0):1, (1,1):1},  # A blocks
            {(0,0):1, (1,1):1},  # B blocks
            {(0,0):1, (1,1):1}   # C blocks
        ),
        # M2 = (A10+A11)(B00)
        (
            {(1,0):1, (1,1):1},
            {(0,0):1},
            {(1,0):1}
        ),
        # M3 = (A00)(B01-B11)
        (
            {(0,0):1},
            {(0,1):1, (1,1):-1},
            {(0,1):1}
        ),
        # M4 = (A11)(B10-B00)
        (
            {(1,1):1},
            {(1,0):1, (0,0):-1},
            {(1,0):1, (0,0):1}
        ),
        # M5 = (A00+A01)(B11)
        (
            {(0,0):1, (0,1):1},
            {(1,1):1},
            {(0,1):1, (0,0):-1}
        ),
        # M6 = (A10-A00)(B00+B01)
        (
            {(1,0):1, (0,0):-1},
            {(0,0):1, (0,1):1},
            {(1,1):1}
        ),
        # M7 = (A11-A10)(B10+B11)
        (
            {(1,1):1, (1,0):-1},
            {(1,0):1, (1,1):1},
            {(0,0):1}
        ),
    ]
    
    # Now construct the full decomposition
    for outer_idx in range(7):
        A_block_coeffs, B_block_coeffs, C_block_coeffs = outer_strassen[outer_idx]
        
        for inner_idx in range(7):
            left_coeffs, right_coeffs, output_coeffs = strassen_2x2[inner_idx]
            
            # Build U vector (length 16)
            u = [0] * 16
            for (bi, bj), block_coeff in A_block_coeffs.items():
                for (li, lj), local_coeff in left_coeffs.items():
                    # Global A index: row = bi*2 + li, col = bj*2 + lj
                    # U index: row * m + col = (bi*2+li) * 4 + (bj*2+lj)
                    u_idx = (bi * 2 + li) * 4 + (bj * 2 + lj)
                    u[u_idx] += block_coeff * local_coeff
            
            # Build V vector (length 16)
            v = [0] * 16
            for (bj, bk), block_coeff in B_block_coeffs.items():
                for (lj, lk), local_coeff in right_coeffs.items():
                    # Global B index: row = bj*2 + lj, col = bk*2 + lk
                    # V index: row * p + col = (bj*2+lj) * 4 + (bk*2+lk)
                    v_idx = (bj * 2 + lj) * 4 + (bk * 2 + lk)
                    v[v_idx] += block_coeff * local_coeff
            
            # Build W vector (length 16) using trace convention W[k*n+i, r]
            w = [0] * 16
            for (bi, bk), block_coeff in C_block_coeffs.items():
                for (ci, cj), local_coeff in output_coeffs.items():
                    # Global C index: row = bi*2 + ci, col = bk*2 + cj
                    # Trace convention: W[k*n + i] where k is col index, i is row index
                    # So W index = col * n + row = (bk*2+cj) * 4 + (bi*2+ci)
                    w_idx = (bk * 2 + cj) * 4 + (bi * 2 + ci)
                    w[w_idx] += block_coeff * local_coeff
            
            U.append(u)
            V.append(v)
            W.append(w)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
