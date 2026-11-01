"""Matrix multiplication 2x2x15 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-53 decomposition for 2x2x15 matrix multiplication.
    
    Uses Strassen's algorithm (rank 7) for each 2x2x2 sub-block (7 blocks),
    and schoolbook (rank 4) for the final 2x2x1 column.
    Total rank = 7*7 + 4 = 53.
    
    Interface: U (n*m, rank) = (4, 53), V (m*p, rank) = (30, 53), W (p*n, rank) = (30, 53)
    W follows trace convention W[k*n+i, r].
    """
    import time
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = 2, 2, 15
    total_rank = 53
    
    U = [[0] * total_rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * total_rank for _ in range(p * n)]  # 30 x 53
    
    # Strassen 2x2x2 decomposition terms (7 terms)
    # Each term has (U_entry, V_entry, W_entry, coeff)
    # U is 4x4 for 2x2, V is 2x2, W is 2x2
    # Strassen formulas for C = A*B where A, B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U entries for each Strassen term (4-element vectors for 2x2 matrix A)
    # A00=idx0, A01=idx1, A10=idx2, A11=idx3
    strassen_U = [
        [1, 0, 0, 1],  # M1: A00+A11
        [0, 0, 1, 1],  # M2: A10+A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00+A01
        [-1, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1], # M7: A01-A11
    ]
    
    # V entries for each Strassen term (4-element vectors for 2x2 matrix B)
    # B00=idx0, B01=idx1, B10=idx2, B11=idx3
    strassen_V = [
        [1, 0, 1, 0],  # M1: B00+B10... wait, let me recheck
    ]
    
    # Let me be more careful with Strassen:
    # Standard Strassen for C = AB, A and B are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(h-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    # C = [[M1+M4-M5+M7, M3+M5],
    #      [M2+M4,         M1-M2+M3+M6]]
    
    # U entries (for A: [a,b,c,d]):
    strassen_U = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    
    # V entries (for B: [e,f,g,h]):
    strassen_V = [
        [1, 0, 1, 0],   # M1: e+g... wait, (e+h)? No: e+h means idx0 and idx3
        [1, 0, 0, 0],   # M2: e
        [0, 1, -1, 0],  # M3: f-g
        [0, -1, 0, 1],  # M4: h-f
        [0, 0, 0, 1],   # M5: h
        [1, 0, 1, 0],   # M6: e+g
        [0, 1, 0, 1],   # M7: f+h
    ]
    
    # Fix M1: (a+d)(e+h), so V for M1 should be [1, 0, 0, 1] (e+h)
    strassen_V = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, -1, 0],  # M3: f-g
        [0, -1, 0, 1],  # M4: h-f
        [0, 0, 0, 1],   # M5: h
        [1, 0, 1, 0],   # M6: e+g
        [0, 1, 0, 1],   # M7: f+h
    ]
    
    # W entries (for C: [C00,C01,C10,C11] in trace order k*n+i, so C00=idx0, C01=idx2, C10=idx1, C11=idx3)
    # Wait, trace convention W[k*n+i]: k is row of C, i is column.
    # C is 2x2, so k in {0,1}, i in {0,1}
    # W[0*2+0] = C00, W[1*2+0] = C10, W[0*2+1] = C01, W[1*2+1] = C11
    # So W index: C00->0, C10->1, C01->2, C11->3
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # W entries for each Strassen term (4-element vectors for 2x2 C in trace order [C00,C10,C01,C11]):
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 1, 0, -1],  # M2: C10 - C11
        [0, 0, 1, 1],   # M3: C01 + C11
        [1, 1, 0, 0],   # M4: C00 + C10
        [-1, 0, 1, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    # Build Strassen block: 7 columns
    def add_strassen_block(U, V, W, col_offset, v_row_base, w_row_base):
        """Add Strassen 2x2x2 block starting at col_offset, with V rows starting at v_row_base and W rows starting at w_row_base."""
        for term in range(7):
            col = col_offset + term
            for ui in range(4):
                U[ui][col] += strassen_U[term][ui]
            for vi in range(4):
                V[v_row_base + vi][col] += strassen_V[term][vi]
            for wi in range(4):
                W[w_row_base + wi][col] += strassen_W[term][wi]
    
    # Build schoolbook 2x2x1 block: 4 columns
    def add_schoolbook_2x2x1(U, V, W, col_offset, v_row_base, w_row_base):
        """Add schoolbook 2x2x1 block. A is 2x2, B is 2x1, C is 2x1."""
        # C00 = A00*B0 + A01*B1
        # C10 = A10*B0 + A11*B1
        # 4 terms
        terms_U = [
            [1, 0, 0, 0],  # A00
            [0, 1, 0, 0],  # A01
            [0, 0, 1, 0],  # A10
            [0, 0, 0, 1],  # A11
        ]
        terms_V = [
            [1, 0],  # B0
            [0, 1],  # B1
            [1, 0],  # B0
            [0, 1],  # B1
        ]
        # C is 2x1, trace order: C00->idx0, C10->idx1
        terms_W = [
            [1, 0],  # C00
            [1, 0],  # C00
            [0, 1],  # C10
            [0, 1],  # C10
        ]
        
        for term in range(4):
            col = col_offset + term
            for ui in range(4):
                U[ui][col] += terms_U[term][ui]
            for vi in range(2):
                V[v_row_base + vi][col] += terms_V[term][vi]
            for wi in range(2):
                W[w_row_base + wi][col] += terms_W[term][wi]
    
    # 7 Strassen blocks for columns 0-13 (blocks of size 2)
    for block in range(7):
        if time.time() > deadline - 10:
            break
        col_offset = block * 7
        # V rows: B is 2x15, for block `block`, columns 2*block and 2*block+1
        # V index for B[j][k] is j*15 + k
        # For this block, k in {2*block, 2*block+1}, j in {0,1}
        # V rows: 0*15 + 2*block, 0*15 + 2*block+1, 1*15 + 2*block, 1*15 + 2*block+1
        v_row_base = 0
        # We need to map local V indices (0,1,2,3) to global V indices
        # Local: [B00, B01, B10, B11] -> [B[0][2b], B[0][2b+1], B[1][2b], B[1][2b+1]]
        # Global V indices: 0*15+2b, 0*15+2b+1, 1*15+2b, 1*15+2b+1
        
        for term in range(7):
            col = col_offset + term
            # U: 4 entries
            for ui in range(4):
                U[ui][col] = strassen_U[term][ui]
            # V: 4 entries mapped to global
            v_local = strassen_V[term]
            # v_local[0] -> B[0][2b], v_local[1] -> B[0][2b+1], v_local[2] -> B[1][2b], v_local[3] -> B[1][2b+1]
            V[0 * 15 + 2 * block][col] = v_local[0]
            V[0 * 15 + 2 * block + 1][col] = v_local[1]
            V[1 * 15 + 2 * block][col] = v_local[2]
            V[1 * 15 + 2 * block + 1][col] = v_local[3]
            # W: 4 entries, C is 2x15, for block `block`, columns 2*block and 2*block+1
            # W index for C[i][k] is k*2 + i
            # Local: [C00, C10, C01, C11] -> [C[0][2b], C[1][2b], C[0][2b+1], C[1][2b+1]]
            w_local = strassen_W[term]
            W[2 * block * 2 + 0][col] = w_local[0]    # C[0][2b]
            W[2 * block * 2 + 1][col] = w_local[1]    # C[1][2b]
            W[2 * block * 2 + 2][col] = w_local[2]    # C[0][2b+1]
            W[2 * block * 2 + 3][col] = w_local[3]    # C[1][2b+1]
    
    # 1 schoolbook block for column 14
    if time.time() < deadline - 10:
        col_offset = 7 * 7  # 49
        # V rows: B[0][14], B[1][14]
        # W rows: C[0][14], C[1][14]
        terms_U = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        terms_V = [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ]
        terms_W = [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ]
        for term in range(4):
            col = col_offset + term
            for ui in range(4):
                U[ui][col] = terms_U[term][ui]
            V[0 * 15 + 14][col] = terms_V[term][0]
            V[1 * 15 + 14][col] = terms_V[term][1]
            W[14 * 2 + 0][col] = terms_W[term][0]
            W[14 * 2 + 1][col] = terms_W[term][1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
