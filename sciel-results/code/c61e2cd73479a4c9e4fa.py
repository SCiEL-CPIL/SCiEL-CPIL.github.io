def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    col = 0
    
    def add_strassen_2x2(a_entries, b_entries, c_coefficients):
        """
        a_entries: 4 lists, each is list of (U_index, coef) for the 4 entries of the 2x2 A block
        b_entries: 4 lists, each is list of (V_index, coef) for the 4 entries of the 2x2 B block
        c_coefficients: dict mapping (c_row, c_col) -> coefficient for the 2x2 C block
        """
        nonlocal col
        # a_entries[0] = entry (0,0), a_entries[1] = entry (0,1), etc.
        a00, a01, a10, a11 = a_entries
        b00, b01, b10, b11 = b_entries
        
        # Strassen formulas:
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)(b00)
        # M3 = (a00)(b01 - b11)
        # M4 = (a11)(b10 - b00)
        # M5 = (a00 + a01)(b11)
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each of the 7 multiplications, add a rank-1 term
        
        # M1 = (a00 + a11)(b00 + b11)
        # C11 += 1*M1, C12 -= 1*M1, C21 -= 1*M1, C22 += 1*M1... wait
        # Actually C11 = M1 + M4 - M5 + M7, so M1 contributes +1 to C11
        # C12 = M3 + M5, so M1 contributes 0 to C12
        # C21 = M2 + M4, so M1 contributes 0 to C21
        # C22 = M1 - M2 + M3 + M6, so M1 contributes +1 to C22
        
        # Wait, I need to double-check Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # M1: C11+=1, C22+=1
        # M2: C21+=1, C22-=1
        # M3: C12+=1, C21+=1, C22+=1  -- wait no
        # Let me recheck:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1: C11+=1, C22+=1
        # M2: C21+=1, C22-=1
        # M3: C12+=1, C22+=1
        # M4: C11+=1, C21+=1
        # M5: C11-=1, C12+=1
        # M6: C22+=1
        # M7: C11+=1
        
        # Hmm wait, that doesn't look right. Let me verify with a simple case.
        # Actually the standard Strassen is:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)(b11)
        # M3 = (a11)(b12-b22)
        # M4 = (a22)(b21-b11)
        # M5 = (a11+a12)(b22)
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # OK so my formulas above are correct.
        
        def add_term(a_terms, b_terms, w_coeffs):
            """Add one rank-1 term."""
            nonlocal col
            for (u_idx, coef) in a_terms:
                U[u_idx][col] += coef
            for (v_idx, coef) in b_terms:
                V[v_idx][col] += coef
            for (c_pos, w_coef) in w_coeffs.items():
                # c_pos is (row_in_c_block, col_in_c_block)
                # We need to map this to the global W index
                # The caller should pass global W indices
                pass
            col += 1
        
        # I'll restructure: pass global W indices directly
        # c_coefficients maps (global_W_index) -> coefficient
        
        # M1 = (a00 + a11)(b00 + b11)
        a_m1 = a00 + a11
        b_m1 = b00 + b11
        # W contributions: C11 += 1, C22 += 1
        for (u_idx, coef) in a_m1:
            U[u_idx][col] += coef
        for (v_idx, coef) in b_m1:
            V[v_idx][col] += coef
        for (w_idx, w_coef) in c_coefficients.get('C11', []):
            W[w_idx][col] += 1 * w_coef
        for (w_idx, w_coef) in c_coefficients.get('C22', []):
            W[w_idx][col] += 1 * w_coef
        col += 1
        
        # M2 = (a10 + a11)(b00)
        a_m2 = a10 + a11
        b_m2 = b00
        for (u_idx, coef) in a_m2:
            U[u_idx][col] += coef
        for (v_idx, coef) in b_m2:
            V[v_idx][col] += coef
        for (w_idx, w_coef) in c_coefficients.get('C21', []):
            W[w_idx][col] += 1 * w_coef
        for (w_idx, w_coef) in c_coefficients.get('C22', []):
            W[w_idx][col] += -1 * w_coef
        col += 1
        
        # M3 = (a00)(b01 - b11)
        a_m3 = a00
        b_m3 = [(v, -c) for (v, c) in b11] + b01
        for (u_idx, coef) in a_m3:
            U[u_idx][col] += coef
        for (v_idx, coef) in b_m3:
            V[v_idx][col] += coef
        for (w_idx, w_coef) in c_coefficients.get('C12', []):
            W[w_idx][col] += 1 * w_coef
        for (w_idx, w_coef) in c_coefficients.get('C22', []):
            W[w_idx][col] += 1 * w_coef
        col += 1
        
        # M4 = (a11)(b10 - b00)
        a_m4 = a11
        b_m4 = [(v, -c) for (v, c) in b00] + b10
        for (u_idx, coef) in a_m4:
            U[u_idx][col] += coef
        for (v_idx, coef) in b_m4:
            V[v_idx][col] += coef
        for (w_idx, w_coef) in c_coefficients.get('C11', []):
            W[w_idx][col] += 1 * w_coef
        for (w_idx, w_coef) in c_coefficients.get('C21', []):
            W[w_idx][col] += 1 * w_coef
        col += 1
        
        # M5 = (a00 + a01)(b11)
        a_m5 = a00 + a01
        b_m5 = b11
        for (u_idx, coef) in a_m5:
            U[u_idx][col] += coef
        for (v_idx, coef) in b_m5:
            V[v_idx][col] += coef
        for (w_idx, w_coef) in c_coefficients.get('C11', []):
            W[w_idx][col] += -1 * w_coef
        for (w_idx, w_coef) in c_coefficients.get('C12', []):
            W[w_idx][col] += 1 * w_coef
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        a_m6 = [(u, c) for (u, c) in a10] + [(u, -c) for (u, c) in a00]
        b_m6 = b00 + b01
        for (u_idx, coef) in a_m6:
            U[u_idx][col] += coef
        for (v_idx, coef) in b_m6:
            V[v_idx][col] += coef
        for (w_idx, w_coef) in c_coefficients.get('C22', []):
            W[w_idx][col] += 1 * w_coef
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        a_m7 = [(u, c) for (u, c) in a01] + [(u, -c) for (u, c) in a11]
        b_m7 = b10 + b11
        for (u_idx, coef) in a_m7:
            U[u_idx][col] += coef
        for (v_idx, coef) in b_m7:
            V[v_idx][col] += coef
        for (w_idx, w_coef) in c_coefficients.get('C11', []):
            W[w_idx][col] += 1 * w_coef
        col += 1
        
        return 7
    
    # Now define the blocks for the 4x4x4 part
    # A block (i,j) means A[i,j] -> U index = i*4 + j
    # B block (j,k) means B[j,k] -> V index = j*5 + k
    # C block (i,k) means C[i,k] -> W index = k*4 + i
    
    def a_idx(i, j):
        return i * m + j
    
    def b_idx(j, k):
        return j * p + k
    
    def w_idx(i, k):
        return k * n + i
    
    # Define the 2x2 blocks for M1 = (A11+A22)(B11+B22)
    # A part:
    # a00 = A[0,0] + A[2,2]
    # a01 = A[0,1] + A[2,3]
    # a10 = A[1,0] + A[3,2]
    # a11 = A[1,1] + A[3,3]
    
    # B part (using B columns 0-3):
    # b00 = B[0,0] + B[2,2]
    # b01 = B[0,1] + B[2,3]
    # b10 = B[1,0] + B[3,2]
    # b11 = B[1,1] + B[3,3]
    
    # C part (output columns 0-3):
    # C11: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices: 0*4+0=0, 1*4+0=4, 0*4+1=1, 1*4+1=5
    # C12: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices: 2*4+0=8, 3*4+0=12, 2*4+1=9, 3*4+1=13
    # C21: C[2,0], C[2,1], C[3,0], C[3,1] -> W indices: 0*4+2=2, 1*4+2=6, 0*4+3=3, 1*4+3=7
    # C22: C[2,2], C[2,3], C[3,2], C[3,3] -> W indices: 2*4+2=10, 3*4+2=14, 2*4+3=11, 3*4+3=15
    
    # For each of the 7 Strassen products at level 1, I need to define the 2x2 blocks
    # and then apply Strassen at level 2.
    
    # This is very complex. Let me take a different approach: I'll directly enumerate
    # all 49 terms by working through the two-level Strassen.
    
    # Actually, let me define a helper that, given the level-1 Strassen product definition,
    # expands it into 7 level-2 Strassen terms.
    
    # Level 1 Strassen products:
    # M1 = (A11+A22)(B11+B22) -> C11, C22
    # M2 = (A21+A22)(B11) -> C21, C22
    # M3 = (A11)(B12-B22) -> C12, C22
    # M4 = (A22)(B21-B11) -> C11, C21
    # M5 = (A11+A12)(B22) -> C11, C12
    # M6 = (A21-A11)(B11+B12) -> C22
    # M7 = (A12-A22)(B21+B22) -> C11
    
    # For each, the "A block" and "B block" are 2x2 matrices whose entries are linear
    # combinations of the original 4x4 entries.
    
    # Let me define each level-1 product's A and B blocks as 2x2 arrays of (index, coef) lists.
    
    def make_block_2x2(entries):
        """entries is a 2x2 list of (global_idx, coef) lists"""
        return entries
    
    # M1: A block = A11 + A22
    # A11 entries: (0,0)->U[0], (0,1)->U[1], (1,0)->U[4], (1,1)->U[5]
    # A22 entries: (2,2)->U[10], (2,3)->U[11], (3,2)->U[14], (3,3)->U[15]
    # So A block for M1:
    #   (0,0): [(0,1), (10,1)]
    #   (0,1): [(1,1), (11,1)]
    #   (1,0): [(4,1), (14,1)]
    #   (1,1): [(5,1), (15,1)]
    
    # B block for M1 = B11 + B22 (B cols 0-3)
    # B11: (0,0)->V[0], (0,1)->V[1], (1,0)->V[5], (1,1)->V[6]
    # B22: (2,2)->V[12], (2,3)->V[13], (3,2)->V[17], (3,3)->V[18]
    # So B block for M1:
    #   (0,0): [(0,1), (12,1)]
    #   (0,1): [(1,1), (13,1)]
    #   (1,0): [(5,1), (17,1)]
    #   (1,1): [(6,1), (18,1)]
    
    # C coefficients for M1: C11 += 1, C22 += 1
    # C11 W indices: [0, 1, 4, 5] (C[0,0], C[0,1], C[1,0], C[1,1])
    # C22 W indices: [10, 11, 14, 15] (C[2,2], C[2,3], C[3,2], C[3,3])
    
    # This is getting very tedious. Let me just write it all out programmatically.
    
    col = 0
    
    # Helper to add Strassen for a 2x2x2 where blocks are specified as 2x2 arrays of (idx, coef) lists
    def add_strassen(a_block, b_block, c_w_indices):
        """
        a_block: 2x2 list, each element is list of (U_idx, coef)
        b_block: 2x2 list, each element is list of (V_idx, coef)
        c_w_indices: dict with keys 'C11','C12','C21','C22', each value is list of W indices
        """
        nonlocal col
        
        a00 = a_block[0][0]
        a01 = a_block[0][1]
        a10 = a_block[1][0]
        a11 = a_block[1][1]
        b00 = b_block[0][0]
        b01 = b_block[0][1]
        b10 = b_block[1][0]
        b11 = b_block[1][1]
        
        C11_w = c_w_indices.get('C11', [])
        C12_w = c_w_indices.get('C12', [])
        C21_w = c_w_indices.get('C21', [])
        C22_w = c_w_indices.get('C22', [])
        
        # M1 = (a00+a11)(b00+b11)
        a_m1 = a00 + a11
        b_m1 = b00 + b11
        for (u, c) in a_m1: U[u][col] += c
        for (v, c) in b_m1: V[v][col] += c
        for w in C11_w: W[w][col] += 1
        for w in C22_w: W[w][col] += 1
        col += 1
        
        # M2 = (a10+a11)(b00)
        a_m2 = a10 + a11
        b_m2 = b00
        for (u, c) in a_m2: U[u][col] += c
        for (v, c) in b_m2: V[v][col] += c
        for w in C21_w: W[w][col] += 1
        for w in C22_w: W[w][col] += -1
        col += 1
        
        # M3 = (a00)(b01-b11)
        a_m3 = a00
        b_m3 = b01 + [(v, -c) for (v, c) in b11]
        for (u, c) in a_m3: U[u][col] += c
        for (v, c) in b_m3: V[v][col] += c
        for w in C12_w: W[w][col] += 1
        for w in C22_w: W[w][col] += 1
        col += 1
        
        # M4 = (a11)(b10-b00)
        a_m4 = a11
        b_m4 = b10 + [(v, -c) for (v, c) in b00]
        for (u, c) in a_m4: U[u][col] += c
        for (v, c) in b_m4: V[v][col] += c
        for w in C11_w: W[w][col] += 1
        for w in C21_w: W[w][col] += 1
        col += 1
        
        # M5 = (a00+a01)(b11)
        a_m5 = a00 + a01
        b_m5 = b11
        for (u, c) in a_m5: U[u][col] += c
        for (v, c) in b_m5: V[v][col] += c
        for w in C11_w: W[w][col] += -1
        for w in C12_w: W[w][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        a_m6 = a10 + [(u, -c) for (u, c) in a00]
        b_m6 = b00 + b01
        for (u, c) in a_m6: U[u][col] += c
        for (v, c) in b_m6: V[v][col] += c
        for w in C22_w: W[w][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        a_m7 = a01 + [(u, -c) for (u, c) in a11]
        b_m7 = b10 + b11
        for (u, c) in a_m7: U[u][col] += c
        for (v, c) in b_m7: V[v][col] += c
        for w in C11_w: W[w][col] += 1
        col += 1
        
        return 7
    
    # Define W index lists for each C block
    # C[i,k] -> W[k*n+i] = W[k*4+i]
    C11_w = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]  # [0, 4, 1, 5]
    C12_w = [w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)]  # [8, 12, 9, 13]
    C21_w = [w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)]  # [2, 6, 3, 7]
    C22_w = [w_idx(2,2), w_idx(2,3), w_idx(3,2), w_idx(3,3)]  # [10, 14, 11, 15]
    
    # Now define the 7 level-1 Strassen products and expand each with level-2 Strassen
    
    # Helper to create a 2x2 block from A entries
    def a_block_from(rows, cols, sign=1):
        """Create a 2x2 block from A[row][col] entries with given sign"""
        block = []
        for r in rows:
            row = []
            for c in cols:
                row.append([(a_idx(r, c), sign)])
            block.append(row)
        return block
    
    def a_block_sum(block1, block2, sign2=1):
        """Sum two 2x2 A blocks"""
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append(block1[i][j] + [(u, c*sign2) for (u, c) in block2[i][j]])
            result.append(row)
        return result
    
    def b_block_from(rows, cols, sign=1):
        """Create a 2x2 block from B[row][col] entries (cols 0-3)"""
        block = []
        for r in rows:
            row = []
            for c in cols:
                row.append([(b_idx(r, c), sign)])
            block.append(row)
        return block
    
    def b_block_sum(block1, block2, sign2=1):
        """Sum two 2x2 B blocks"""
        result = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append(block1[i][j] + [(v, c*sign2) for (v, c) in block2[i][j]])
            result.append(row)
        return result
    
    # A blocks:
    A11 = a_block_from([0,1], [0,1])
    A12 = a_block_from([0,1], [2,3])
    A21 = a_block_from([2,3], [0,1])
    A22 = a_block_from([2,3], [2,3])
    
    # B blocks (cols 0-3):
    B11 = b_block_from([0,1], [0,1])
    B12 = b_block_from([0,1], [2,3])
    B21 = b_block_from([2,3], [0,1])
    B22 = b_block_from([2,3], [2,3])
    
    # C W index dicts:
    c_w_M1 = {'C11': C11_w, 'C22': C22_w}
    c_w_M2 = {'C21': C21_w, 'C22': C22_w}
    c_w_M3 = {'C12': C12_w, 'C22': C22_w}
    c_w_M4 = {'C11': C11_w, 'C21': C21_w}
    c_w_M5 = {'C11': C11_w, 'C12': C12_w}
    c_w_M6 = {'C22': C22_w}
    c_w_M7 = {'C11': C11_w}
    
    # M1 = (A11+A22)(B11+B22)
    add_strassen(a_block_sum(A11, A22), b_block_sum(B11, B22), c_w_M1)
    
    # M2 = (A21+A22)(B11)
    add_strassen(a_block_sum(A21, A22), B11, c_w_M2)
    
    # M3 = (A11)(B12-B22)
    add_strassen(A11, b_block_sum(B12, B22, -1), c_w_M3)
    
    # M4 = (A22)(B21-B11)
    add_strassen(A22, b_block_sum(B21, B11, -1), c_w_M4)
    
    # M5 = (A11+A12)(B22)
    add_strassen(a_block_sum(A11, A12), B22, c_w_M5)
    
    # M6 = (A21-A11)(B11+B12)
    add_strassen(a_block_sum(A21, A11, -1), b_block_sum(B11, B12), c_w_M6)
    
    # M7 = (A12-A22)(B21+B22)
    add_strassen(a_block_sum(A12, A22, -1), b_block_sum(B21, B22), c_w_M7)
    
    # Now Part 2: 4x4 * 4x1 (B column 4) using schoolbook
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1  # W[k*n+i] with k=4
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
