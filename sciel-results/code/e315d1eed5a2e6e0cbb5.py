"""Matrix multiplication 4x4 by 4x9 tensor decomposition.

Strategy:
  - Split the 4x9 matrix B into three parts: B1 (cols 0-3), B2 (cols 4-7), B3 (col 8).
  - For B1 and B2 (each 4x4), use a rank-49 Strassen-based decomposition.
  - For B3 (4x1), use a rank-16 schoolbook decomposition.
  - Total rank = 49 + 49 + 16 = 114.

The output tensor uses the trace convention W[k*n + i, r] where n=4 (rows of A / cols of B).
"""

DIMENSIONS = [4, 4, 9]


def _strassen_4x4x4():
    """
    Generate a rank-49 decomposition for 4x4 * 4x4 -> 4x4 matrix multiplication.
    
    Uses two levels of Strassen:
      - Outer: decompose each 4x4 into 2x2 blocks -> 7 multiplications of 2x2*2x2
      - Inner: each 2x2*2x2 -> 2x2 uses 7 rank-1 terms (Strassen)
    
    Returns lists U_rows, V_rows, W_rows where:
      U_rows: list of (index_in_16, coeff) for each of the 49 terms
      V_rows: list of (index_in_16, coeff) for each of the 49 terms
      W_rows: list of (index_in_16, coeff) for each of the 49 terms
    """
    # Inner Strassen for 2x2 * 2x2 -> 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each of the 7 inner multiplications, we have:
    # A_block = sum of coeff * A[i,j] for (i,j) in {(0,0),(0,1),(1,0),(1,1)}
    # B_block = sum of coeff * B[j,k] for (j,k) in {(0,0),(0,1),(1,0),(1,1)}
    # C_block[i,k] += coeff * (A_block * B_block)
    
    # Inner Strassen terms: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from (i,j) to coeff for the 2x2 A block
    # b_coeffs: dict from (j,k) to coeff for the 2x2 B block
    # c_coeffs: dict from (i,k) to coeff for the 2x2 C block
    
    inner_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M4
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 0, (1,0): 0, (1,1): 0}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Outer Strassen for 4x4 * 4x4 -> 4x4
    # Divide into 2x2 blocks of 2x2 submatrices.
    # Let A = [[A00, A01], [A10, A11]] where each is 2x2
    # Similarly B and C.
    
    # Outer Strassen terms:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B10-B11)
    # S4 = A11(B01-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    # C00 = S1+S4-S5+S7
    # C01 = S3+S5
    # C10 = S2+S4
    # C11 = S1-S2+S3+S6
    
    # For each outer term, we combine inner terms.
    # Each outer term involves a linear combination of A blocks and B blocks.
    # The A block combination is a 2x2 matrix of linear combinations of scalar A entries.
    # Similarly for B.
    
    # We'll build the 49 rank-1 terms by iterating over outer and inner.
    
    U_terms = []  # list of (u_index, coeff)
    V_terms = []  # list of (v_index, coeff)
    W_terms = []  # list of (w_index, coeff)
    
    # Outer Strassen definitions:
    # Each outer term: (a_block_coeffs, b_block_coeffs, c_block_coeffs)
    # a_block_coeffs: dict from (block_i, block_j) to coeff, where block_i, block_j in {0,1}
    #   This means A_block[i,j] = sum of coeff * A[2*i+a, 2*j+b] for a,b in {0,1}
    #   Actually, it's simpler: a_block_coeffs maps (bi, bj) -> coeff, meaning
    #   the combined A 2x2-block has entry (a,b) = sum over (bi,bj) of coeff * A[2*bi+a, 2*bj+b]
    
    # Let me think more carefully.
    # For outer S1: A_combined = A00 + A11, B_combined = B00 + B11
    # A00 is the 2x2 block at rows 0-1, cols 0-1.
    # A11 is the 2x2 block at rows 2-3, cols 2-3.
    # So A_combined[a,b] = A[2*0+a, 2*0+b] + A[2*1+a, 2*1+b] for a,b in {0,1}
    # = A[a, b] + A[2+a, 2+b]
    
    # So for outer S1, the A 2x2-block has:
    #   entry (a,b) = sum over (bi,bj) of coeff(bi,bj) * A[2*bi+a, 2*bj+b]
    # where coeff(0,0)=1, coeff(1,1)=1, others=0.
    
    # Similarly for B.
    
    outer_terms = [
        # S1: (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        # S2: (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
        # S3: A00(B10-B11)
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
        # S4: A11(B01-B00)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        # S5: (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 0, (1,0): 0, (1,1): 0}),
        # S6: (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # S7: (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    for outer_a, outer_b, outer_c in outer_terms:
        for inner_a, inner_b, inner_c in inner_terms:
            # Build the scalar coefficients for A, B, C.
            # A scalar: sum over (bi,bj) in outer_a and (a,b) in inner_a
            #   coeff = outer_a[(bi,bj)] * inner_a[(a,b)]
            #   A index = 2*bi + a, 2*bj + b
            #   linear index in U = (2*bi+a)*4 + (2*bj+b)
            
            a_coeffs = {}
            for (bi, bj), ca in outer_a.items():
                for (a, b), ci in inner_a.items():
                    idx = (2*bi + a) * 4 + (2*bj + b)
                    a_coeffs[idx] = a_coeffs.get(idx, 0) + ca * ci
            
            # B scalar: sum over (bi,bj) in outer_b and (a,b) in inner_b
            #   B block at (bi,bj) has entries B[2*bi+a, 2*bj+b]
            #   inner_b maps (a,b) -> coeff, where a is row within 2x2, b is col within 2x2
            #   So B index = 2*bi + a, 2*bj + b
            #   linear index in V = (2*bi+a)*4 + (2*bj+b)
            
            b_coeffs = {}
            for (bi, bj), cb in outer_b.items():
                for (a, b), ci in inner_b.items():
                    idx = (2*bi + a) * 4 + (2*bj + b)
                    b_coeffs[idx] = b_coeffs.get(idx, 0) + cb * ci
            
            # C scalar: sum over (bi,bj) in outer_c and (a,b) in inner_c
            #   C block at (bi,bj) has entries C[2*bi+a, 2*bj+b]
            #   inner_c maps (a,b) -> coeff
            #   So C index = 2*bi + a, 2*bj + b
            #   linear index in W = (2*bj+b)*4 + (2*bi+a)  [W[k*n+i] where k=col, i=row]
            
            c_coeffs = {}
            for (bi, bj), cc in outer_c.items():
                for (a, b), ci in inner_c.items():
                    row = 2*bi + a
                    col = 2*bj + b
                    idx = col * 4 + row
                    c_coeffs[idx] = c_coeffs.get(idx, 0) + cc * ci
            
            U_terms.append(a_coeffs)
            V_terms.append(b_coeffs)
            W_terms.append(c_coeffs)
    
    return U_terms, V_terms, W_terms


def construct():
    """
    Construct the tensor decomposition for 4x4 * 4x9 -> 4x9 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (16, rank)
    V: shape (36, rank)
    W: shape (36, rank)
    """
    import time
    start_time = time.time()
    deadline = 150.0  # seconds
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Get the 4x4x4 Strassen decomposition
    U_4x4, V_4x4, W_4x4 = _strassen_4x4x4()
    
    rank = len(U_4x4) * 2 + 16  # 49*2 + 16 = 114
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 36 x rank
    W = [[0] * rank for _ in range(p * n)]  # 36 x rank
    
    col = 0
    
    # Part 1: A * B1 where B1 is cols 0-3 of B
    for u_coeffs, v_coeffs, w_coeffs in zip(U_4x4, V_4x4, W_4x4):
        for idx, coeff in u_coeffs.items():
            U[idx][col] += coeff
        for idx, coeff in v_coeffs.items():
            V[idx][col] += coeff  # B1 cols 0-3, so V index is same
        for idx, coeff in w_coeffs.items():
            W[idx][col] += coeff
        col += 1
    
    # Part 2: A * B2 where B2 is cols 4-7 of B
    for u_coeffs, v_coeffs, w_coeffs in zip(U_4x4, V_4x4, W_4x4):
        for idx, coeff in u_coeffs.items():
            U[idx][col] += coeff
        for idx, coeff in v_coeffs.items():
            # B2 is cols 4-7, so V index = row*9 + (4 + col_offset)
            # Original idx is for a 4x4 B block, so idx = row*4 + col_offset
            row = idx // 4
            col_offset = idx % 4
            new_idx = row * 9 + (4 + col_offset)
            V[new_idx][col] += coeff
        for idx, coeff in w_coeffs.items():
            # W index for C cols 4-7: col = 4 + original_col, row = original_row
            orig_col = idx // 4
            orig_row = idx % 4
            new_idx = (4 + orig_col) * 4 + orig_row
            W[new_idx][col] += coeff
        col += 1
    
    # Part 3: A * B3 where B3 is col 8 of B (4x1 schoolbook)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1  # j*9 + 8
            W[8 * n + i][col] = 1  # 8*4 + i
            col += 1
    
    # Verify rank is as expected
    assert col == rank, f"Rank mismatch: col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
