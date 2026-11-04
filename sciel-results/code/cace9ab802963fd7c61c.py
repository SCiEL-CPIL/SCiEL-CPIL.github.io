import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 4, 4, 5
    rank = 65  # 49 (Strassen 4x4x4) + 16 (4x4x1)

    U = [[0] * rank for _ in range(n * m)]  # (16, 65)
    V = [[0] * rank for _ in range(m * p)]  # (20, 65)
    W = [[0] * rank for _ in range(p * n)]  # (20, 65)

    col = 0

    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term.
        u_entries: list of (row_idx, coeff) for U
        v_entries: list of (row_idx, coeff) for V
        w_entries: list of (row_idx, coeff) for W
        """
        nonlocal col
        for idx, coeff in u_entries:
            U[idx][col] = coeff
        for idx, coeff in v_entries:
            V[idx][col] = coeff
        for idx, coeff in w_entries:
            W[idx][col] = coeff
        col += 1

    # Strassen 2x2x2 base case: 7 rank-1 terms
    # For multiplying 2x2 X * Y = Z:
    # P1 = (x00+x11)(y00+y11)
    # P2 = (x10+x11)y00
    # P3 = x00(y10-y11)
    # P4 = x11(y01-y00)
    # P5 = (x00+x01)y11
    # P6 = (x10-x00)(y00+y01)
    # P7 = (x01-x11)(y10+y11)
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6

    # The 7 terms for 2x2x2, each with:
    # u: entries in X (2x2), v: entries in Y (2x2), w: entries in Z (2x2)
    # X indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Y indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Z indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3

    strassen_2x2_terms = [
        # (u_entries, v_entries, w_entries)
        # P1: (x00+x11)(y00+y11) -> Z00, Z11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # P2: (x10+x11)y00 -> Z10
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # P3: x00(y10-y11) -> Z01
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1)]),
        # P4: x11(y01-y00) -> Z00, Z10
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
        # P5: (x00+x01)y11 -> Z00, Z01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # P6: (x10-x00)(y00+y01) -> Z11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # P7: (x01-x11)(y10+y11) -> Z00
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    # Strassen 4x4x4: 7 products of 2x2 blocks
    # A blocks: A00=[0,1,4,5], A01=[2,3,6,7], A10=[8,9,12,13], A11=[10,11,14,15]
    # B blocks: B00=[0,1,5,6], B01=[2,3,7,8], B10=[10,11,15,16], B11=[12,13,17,18]
    # C blocks: C00=[0,1,4,5], C01=[2,3,6,7], C10=[8,9,12,13], C11=[10,11,14,15]

    # Block index mapping:
    # A block (bi, bj) where bi=block_row (0 or 1), bj=block_col (0 or 1)
    # Global A index for element (i,j) in block (bi,bj): (bi*2+i)*4 + (bj*2+j)
    # B block (bi, bj): global B index for element (i,j) in block (bi,bj): (bi*2+i)*5 + (bj*2+j)
    # C block (bi, bj): global C index for element (i,j) in block (bi,bj): (bj*2+i)*4 + (bi*2+j)
    # Wait, C has shape (p*n) = (5*4) = 20, indexed as k*n+i where k is column of C, i is row.
    # C[i,k] where i=0..3, k=0..3 for the 4x4 part.
    # W index for C[i,k]: k*n + i = k*4 + i

    def a_global(bi, bj, i, j):
        """Global U index for A element at block (bi,bj), local (i,j)."""
        return (bi * 2 + i) * 4 + (bj * 2 + j)

    def b_global(bi, bj, i, j):
        """Global V index for B element at block (bi,bj), local (i,j)."""
        return (bi * 2 + i) * 5 + (bj * 2 + j)

    def w_global(ci, cj, i, j):
        """Global W index for C element at block (ci,cj), local (i,j).
        ci = block column of C (0 or 1), cj = block row of C (0 or 1)
        C[i,k] where i is row, k is column.
        Block (cj, ci) means rows cj*2..cj*2+1, cols ci*2..ci*2+1
        Local (i,j) means row cj*2+i, col ci*2+j
        W index = k*n + i = (ci*2+j)*4 + (cj*2+i)
        """
        return (ci * 2 + j) * 4 + (cj * 2 + i)

    # Strassen 4x4x4: 7 products
    # M1 = (A00 + A11)(B00 + B11) -> contributes to C00, C11
    # M2 = (A10 + A11)B00 -> contributes to C10
    # M3 = A00(B10 - B11) -> contributes to C01
    # M4 = A11(B01 - B00) -> contributes to C00, C10
    # M5 = (A00 + A01)B11 -> contributes to C00, C01
    # M6 = (A10 - A00)(B00 + B01) -> contributes to C11
    # M7 = (A01 - A11)(B10 + B11) -> contributes to C00

    # For each Strassen product, we need to expand the 2x2 multiplication into 7 rank-1 terms.
    # The X matrix (from A side) has entries that are linear combinations of A block elements.
    # The Y matrix (from B side) has entries that are linear combinations of B block elements.
    # The Z matrix (output) maps to specific C block elements.

    # Define the 7 Strassen products for 4x4x4:
    # Each product: (X_coeffs, Y_coeffs, Z_coeffs)
    # X_coeffs: dict {(local_i, local_j): (a_block_i, a_block_j, coeff)}
    # Y_coeffs: dict {(local_i, local_j): (b_block_i, b_block_j, coeff)}
    # Z_coeffs: dict {(local_i, local_j): (c_block_i, c_block_j, coeff)}

    # Product 1: M1 = (A00+A11)(B00+B11)
    # X: (0,0)->A00, (1,1)->A11
    # Y: (0,0)->B00, (1,1)->B11
    # Z: (0,0)->C00, (1,1)->C11
    # C00 = Z00, C11 = Z11 (from the 2x2 result)

    # Product 2: M2 = (A10+A11)B00
    # X: (1,0)->A10, (1,1)->A11
    # Y: (0,0)->B00
    # Z: (1,0)->C10

    # Product 3: M3 = A00(B10-B11)
    # X: (0,0)->A00
    # Y: (1,0)->B10, (1,1)->-B11
    # Z: (0,1)->C01

    # Product 4: M4 = A11(B01-B00)
    # X: (1,1)->A11
    # Y: (0,1)->B01, (0,0)->-B00
    # Z: (0,0)->C00, (1,0)->C10

    # Product 5: M5 = (A00+A01)B11
    # X: (0,0)->A00, (0,1)->A01
    # Y: (1,1)->B11
    # Z: (0,0)->C00, (0,1)->C01

    # Product 6: M6 = (A10-A00)(B00+B01)
    # X: (1,0)->A10, (0,0)->-A00
    # Y: (0,0)->B00, (0,1)->B01
    # Z: (1,1)->C11

    # Product 7: M7 = (A01-A11)(B10+B11)
    # X: (0,1)->A01, (1,1)->-A11
    # Y: (1,0)->B10, (1,1)->B11
    # Z: (0,0)->C00

    # For each of the 7 products, expand using Strassen 2x2 (7 terms each) = 49 terms total

    products = [
        # (X_coeffs, Y_coeffs, Z_coeffs)
        # X_coeffs: {local_pos: [(block_i, block_j, coeff), ...]}
        # Y_coeffs: {local_pos: [(block_i, block_j, coeff), ...]}
        # Z_coeffs: {local_pos: [(block_i, block_j, coeff), ...]}

        # P1: (A00+A11)(B00+B11) -> C00, C11
        ({(0,0): [(0,0,1)], (1,1): [(1,1,1)]},
         {(0,0): [(0,0,1)], (1,1): [(1,1,1)]},
         {(0,0): [(0,0,1)], (1,1): [(1,1,1)]}),

        # P2: (A10+A11)B00 -> C10
        ({(1,0): [(1,0,1)], (1,1): [(1,1,1)]},
         {(0,0): [(0,0,1)]},
         {(1,0): [(1,0,1)]}),

        # P3: A00(B10-B11) -> C01
        ({(0,0): [(0,0,1)]},
         {(1,0): [(1,0,1)], (1,1): [(1,1,-1)]},
         {(0,1): [(0,1,1)]}),

        # P4: A11(B01-B00) -> C00, C10
        ({(1,1): [(1,1,1)]},
         {(0,1): [(0,1,1)], (0,0): [(0,0,-1)]},
         {(0,0): [(0,0,1)], (1,0): [(1,0,1)]}),

        # P5: (A00+A01)B11 -> C00, C01
        ({(0,0): [(0,0,1)], (0,1): [(0,1,1)]},
         {(1,1): [(1,1,1)]},
         {(0,0): [(0,0,1)], (0,1): [(0,1,1)]}),

        # P6: (A10-A00)(B00+B01) -> C11
        ({(1,0): [(1,0,1)], (0,0): [(0,0,-1)]},
         {(0,0): [(0,0,1)], (0,1): [(0,1,1)]},
         {(1,1): [(1,1,1)]}),

        # P7: (A01-A11)(B10+B11) -> C00
        ({(0,1): [(0,1,1)], (1,1): [(1,1,-1)]},
         {(1,0): [(1,0,1)], (1,1): [(1,1,1)]},
         {(0,0): [(0,0,1)]}),
    ]

    # For each product, expand using Strassen 2x2 terms
    for x_coeffs, y_coeffs, z_coeffs in products:
        for u_2x2, v_2x2, w_2x2 in strassen_2x2_terms:
            # Build global U, V, W entries
            u_entries = []
            v_entries = []
            w_entries = []

            # Process U entries
            for local_pos, coeff in u_2x2:
                li, lj = local_pos
                if (li, lj) in x_coeffs:
                    for (bi, bj, c) in x_coeffs[(li, lj)]:
                        gidx = a_global(bi, bj, li, lj)
                        u_entries.append((gidx, coeff * c))

            # Process V entries
            for local_pos, coeff in v_2x2:
                li, lj = local_pos
                if (li, lj) in y_coeffs:
                    for (bi, bj, c) in y_coeffs[(li, lj)]:
                        gidx = b_global(bi, bj, li, lj)
                        v_entries.append((gidx, coeff * c))

            # Process W entries
            for local_pos, coeff in w_2x2:
                li, lj = local_pos
                if (li, lj) in z_coeffs:
                    for (ci, cj, c) in z_coeffs[(li, lj)]:
                        gidx = w_global(ci, cj, li, lj)
                        w_entries.append((gidx, coeff * c))

            # Combine duplicate entries
            def combine(entries):
                d = {}
                for idx, c in entries:
                    d[idx] = d.get(idx, 0) + c
                return [(idx, c) for idx, c in d.items() if c != 0]

            u_entries = combine(u_entries)
            v_entries = combine(v_entries)
            w_entries = combine(w_entries)

            add_term(u_entries, v_entries, w_entries)

    # Now add the 4x4x1 part for the 5th column (k=4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # This is 16 rank-1 terms (schoolbook)
    for i in range(4):
        for j in range(4):
            add_term(
                [(i * 4 + j, 1)],       # U[i*m+j]
                [(j * 5 + 4, 1)],       # V[j*p+4]
                [(4 * 4 + i, 1)]        # W[4*n+i]
            )

    return {"U": U, "V": V, "W": W}
