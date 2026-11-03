"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 114 decomposition:
    # Two M_4,4,4 blocks (rank 49 each) using recursive Strassen
    # One M_4,4,1 block (rank 16) using schoolbook
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 component
    def add_rank1(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Helper to add Strassen 2x2x2 decomposition
    # A_idx: dict {'a': idx, 'b': idx, 'c': idx, 'd': idx}
    # B_idx: dict {'e': idx, 'f': idx, 'g': idx, 'h': idx}
    # C_idx: dict {'c00': idx, 'c01': idx, 'c10': idx, 'c11': idx}
    def add_strassen_2x2(A_idx, B_idx, C_idx):
        nonlocal col
        
        # M1 = (a+d)(e+h)
        # U: a, d. V: e, h. W: c00(+1), c10(+1)
        # We need to handle the linear combinations.
        # This function assumes A_idx and B_idx point to single indices.
        # But in recursive Strassen, 'a' might be A00_00 + A11_00.
        # So we pass lists of (index, coeff) or just accumulate in U/V/W directly?
        # It's easier to pass the logical indices and let the caller handle the expansion?
        # No, the caller (recursive step) knows the linear combos.
        
        # Let's redefine: This function adds 7 components corresponding to the 7 Strassen multiplications.
        # The inputs A_idx and B_idx should be dictionaries mapping 'a','b','c','d' to 
        # a list of (flat_index, coeff) pairs representing the linear combination.
        
        # M1 = (a+d)(e+h)
        # We need to add a rank-1 tensor for each product term?
        # No, M1 is a single product of two linear combinations.
        # (a+d) is a vector in U-space? No, 'a' and 'd' are elements of the 2x2 matrix.
        # In the 2x2x2 tensor, U has 4 entries, V has 4, W has 4.
        # For M1, U[0] += 1 (for a), U[3] += 1 (for d).
        # V[0] += 1 (for e), V[3] += 1 (for h).
        # W[0] += 1 (for c00), W[2] += 1 (for c10).
        
        # So we just add to the current column.
        
        # M1
        for idx, coeff in A_idx['a']:
            U[idx][col] += coeff
        for idx, coeff in A_idx['d']:
            U[idx][col] += coeff
        for idx, coeff in B_idx['e']:
            V[idx][col] += coeff
        for idx, coeff in B_idx['h']:
            V[idx][col] += coeff
        for idx, coeff in C_idx['c00']:
            W[idx][col] += coeff
        for idx, coeff in C_idx['c10']:
            W[idx][col] += coeff
        col += 1
        
        # M2 = (c+d)e
        for idx, coeff in A_idx['c']:
            U[idx][col] += coeff
        for idx, coeff in A_idx['d']:
            U[idx][col] += coeff
        for idx, coeff in B_idx['e']:
            V[idx][col] += coeff
        for idx, coeff in C_idx['c00']:
            W[idx][col] -= coeff
        for idx, coeff in C_idx['c11']:
            W[idx][col] += coeff
        col += 1
        
        # M3 = a(f-h)
        for idx, coeff in A_idx['a']:
            U[idx][col] += coeff
        for idx, coeff in B_idx['f']:
            V[idx][col] += coeff
        for idx, coeff in B_idx['h']:
            V[idx][col] -= coeff
        for idx, coeff in C_idx['c01']:
            W[idx][col] += coeff
        col += 1
        
        # M4 = d(g-f)
        for idx, coeff in A_idx['d']:
            U[idx][col] += coeff
        for idx, coeff in B_idx['g']:
            V[idx][col] += coeff
        for idx, coeff in B_idx['f']:
            V[idx][col] -= coeff
        for idx, coeff in C_idx['c00']:
            W[idx][col] += coeff
        col += 1
        
        # M5 = (a+b)h
        for idx, coeff in A_idx['a']:
            U[idx][col] += coeff
        for idx, coeff in A_idx['b']:
            U[idx][col] += coeff
        for idx, coeff in B_idx['h']:
            V[idx][col] += coeff
        for idx, coeff in C_idx['c10']:
            W[idx][col] += coeff
        for idx, coeff in C_idx['c11']:
            W[idx][col] -= coeff
        col += 1
        
        # M6 = (c-d)(e+f)
        for idx, coeff in A_idx['c']:
            U[idx][col] += coeff
        for idx, coeff in A_idx['d']:
            U[idx][col] -= coeff
        for idx, coeff in B_idx['e']:
            V[idx][col] += coeff
        for idx, coeff in B_idx['f']:
            V[idx][col] += coeff
        for idx, coeff in C_idx['c11']:
            W[idx][col] += coeff
        col += 1
        
        # M7 = (b-c)(g+h)
        for idx, coeff in A_idx['b']:
            U[idx][col] += coeff
        for idx, coeff in A_idx['c']:
            U[idx][col] -= coeff
        for idx, coeff in B_idx['g']:
            V[idx][col] += coeff
        for idx, coeff in B_idx['h']:
            V[idx][col] += coeff
        for idx, coeff in C_idx['c00']:
            W[idx][col] -= coeff
        for idx, coeff in C_idx['c01']:
            W[idx][col] += coeff
        col += 1

    # Define indices for 4x4x4 block 1 (cols 0-3 of B, cols 0-3 of C)
    # A is shared.
    # A indices:
    # A00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A01: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # A10: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # A11: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15
    
    # B1 (cols 0-3):
    # B00: (0,0)->0, (0,1)->1, (1,0)->9, (1,1)->10
    # B01: (0,2)->2, (0,3)->3, (1,2)->11, (1,3)->12
    # B10: (2,0)->18, (2,1)->19, (3,0)->27, (3,1)->28
    # B11: (2,2)->20, (2,3)->21, (3,2)->29, (3,3)->30
    
    # C1 (cols 0-3):
    # C00: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # C01: (0,2)->2, (0,3)->3, (1,2)->6, (1,3)->7
    # C10: (2,0)->8, (2,1)->9, (3,0)->12, (3,1)->13
    # C11: (2,2)->10, (2,3)->11, (3,2)->14, (3,3)->15

    # Helper to create the index dicts for a 2x2 block
    def get_block_indices(base_a, base_b, base_c, offset_a, offset_b, offset_c):
        # This is getting complex. Let's just hardcode the 7 calls for the top level.
        pass

    # We will manually construct the 7 calls for the first 4x4x4 block.
    # A_idx for top level Strassen:
    # a = A00_00 + A11_00 -> [(0,1), (10,1)]
    # b = A00_01 + A11_01 -> [(1,1), (11,1)]
    # c = A00_10 + A11_10 -> [(4,1), (14,1)]
    # d = A00_11 + A11_11 -> [(5,1), (15,1)]
    
    # Wait, Strassen's a,b,c,d refer to the 2x2 blocks of A.
    # A = [[A00, A01], [A10, A11]]
    # In the formula M1 = (a+d)(e+h), 'a' is the top-left block A00?
    # No, in Strassen's 2x2x2, a,b,c,d are scalars.
    # In the recursive step for 4x4x4, the "scalars" are 2x2 matrices.
    # So 'a' corresponds to the matrix A00.
    # 'd' corresponds to the matrix A11.
    # So for M1, we need to multiply (A00 + A11) and (B00 + B11).
    # This product is a 2x2 matrix multiplication.
    # We decompose this 2x2 matrix multiplication into 7 rank-1 tensors using the inner Strassen.
    
    # So, for each top-level term, we call add_strassen_2x2.
    # The A_idx for the inner call should map 'a','b','c','d' to the elements of the matrix (A00+A11).
    # (A00+A11) is a 2x2 matrix.
    # Its top-left element is A00_00 + A11_00.
    # Its top-right element is A00_01 + A11_01.
    # Its bottom-left element is A00_10 + A11_10.
    # Its bottom-right element is A00_11 + A11_11.
    
    # So for the inner call, A_idx['a'] = [(0,1), (10,1)], etc.
    
    # Let's define the index lists for the 4x4 blocks.
    
    # A Blocks
    A00 = {'a': [(0,1)], 'b': [(1,1)], 'c': [(4,1)], 'd': [(5,1)]}
    A01 = {'a': [(2,1)], 'b': [(3,1)], 'c': [(6,1)], 'd': [(7,1)]}
    A10 = {'a': [(8,1)], 'b': [(9,1)], 'c': [(12,1)], 'd': [(13,1)]}
    A11 = {'a': [(10,1)], 'b': [(11,1)], 'c': [(14,1)], 'd': [(15,1)]}
    
    # B1 Blocks (cols 0-3)
    B00 = {'e': [(0,1)], 'f': [(1,1)], 'g': [(9,1)], 'h': [(10,1)]}
    B01 = {'e': [(2,1)], 'f': [(3,1)], 'g': [(11,1)], 'h': [(12,1)]}
    B10 = {'e': [(18,1)], 'f': [(19,1)], 'g': [(27,1)], 'h': [(28,1)]}
    B11 = {'e': [(20,1)], 'f': [(21,1)], 'g': [(29,1)], 'h': [(30,1)]}
    
    # C1 Blocks (cols 0-3)
    C00 = {'c00': [(0,1)], 'c01': [(1,1)], 'c10': [(4,1)], 'c11': [(5,1)]}
    C01 = {'c00': [(2,1)], 'c01': [(3,1)], 'c10': [(6,1)], 'c11': [(7,1)]}
    C10 = {'c00': [(8,1)], 'c01': [(9,1)], 'c10': [(12,1)], 'c11': [(13,1)]}
    C11 = {'c00': [(10,1)], 'c01': [(11,1)], 'c10': [(14,1)], 'c11': [(15,1)]}
    
    # Function to combine two block index dicts (add coefficients)
    def combine_blocks(b1, b2):
        res = {}
        for key in ['a','b','c','d','e','f','g','h','c00','c01','c10','c11']:
            if key in b1 or key in b2:
                res[key] = []
                if key in b1:
                    res[key].extend(b1[key])
                if key in b2:
                    res[key].extend(b2[key])
        return res
    
    # Top Level Strassen for 4x4x4 Block 1
    # M1 = (A00+A11)(B00+B11) -> C00, C10
    A_M1 = combine_blocks(A00, A11)
    B_M1 = combine_blocks(B00, B11)
    C_M1 = combine_blocks(C00, C10) # Wait, C indices for M1 are c00 and c10 of the 4x4 result.
    # In the inner Strassen call, 'c00' corresponds to the top-left of the 2x2 result.
    # The 2x2 result is the 2x2 block of the 4x4 C.
    # So for M1, the output goes to C00 (top-left 2x2) and C10 (bottom-left 2x2).
    # The inner Strassen produces c00, c01, c10, c11 of the 2x2 product.
    # We need to map these to the correct positions in the 4x4 C.
    # M1 contributes to C[0:2, 0:2] and C[2:4, 0:2].
    # So inner c00 -> C00, inner c10 -> C10.
    # Inner c01 -> C01? No, M1 is (a+d)(e+h).
    # C00 = M1 - M2 + M4 - M7
    # C01 = M3 + M7
    # C10 = M1 + M5
    # C11 = M2 - M5 + M6 + M7
    
    # So for M1, the output is added to C00 (inner c00) and C10 (inner c10).
    # But the inner Strassen call writes to specific C indices.
    # We must pass the correct C indices to the inner call.
    
    # Let's refine add_strassen_2x2 to take explicit C mapping.
    # The C_idx dict should map 'c00' to the list of indices in the global W.
    
    # For M1:
    # Inner c00 -> Global C00
    # Inner c10 -> Global C10
    # Inner c01 -> Unused (0)
    # Inner c11 -> Unused (0)
    
    C_M1 = {
        'c00': C00['c00'],
        'c01': [],
        'c10': C10['c00'],
        'c11': []
    }
    
    add_strassen_2x2(A_M1, B_M1, C_M1)
    
    # M2 = (A10+A11)B00 -> C00, C11
    A_M2 = combine_blocks(A10, A11)
    B_M2 = B00
    C_M2 = {
        'c00': C00['c00'], # Coeff -1
        'c01': [],
        'c10': [],
        'c11': C11['c00'] # Coeff +1
    }
    # Note: add_strassen_2x2 handles signs internally for M2?
    # M2 = (c+d)e.
    # Inner c00 gets -1, inner c11 gets +1.
    # So we just pass the indices.
    add_strassen_2x2(A_M2, B_M2, C_M2)
    
    # M3 = A00(B01-B11) -> C01
    A_M3 = A00
    B_M3 = combine_blocks(B01, B11)
    # B_M3 needs to represent (B01 - B11).
    # So we need to negate B11 coefficients.
    B_M3_neg = {
        'e': [(idx, -coeff) for idx, coeff in B11['e']],
        'f': [(idx, -coeff) for idx, coeff in B11['f']],
        'g': [(idx, -coeff) for idx, coeff in B11['g']],
        'h': [(idx, -coeff) for idx, coeff in B11['h']]
    }
    B_M3 = combine_blocks(B01, B_M3_neg)
    C_M3 = {
        'c00': [],
        'c01': C01['c00'], # Coeff +1
        'c10': [],
        'c11': []
    }
    add_strassen_2x2(A_M3, B_M3, C_M3)
    
    # M4 = A11(B10-B01) -> C00
    A_M4 = A11
    B_M4_neg = {
        'e': [(idx, -coeff) for idx, coeff in B01['e']],
        'f': [(idx, -coeff) for idx, coeff in B01['f']],
        'g': [(idx, -coeff) for idx, coeff in B01['g']],
        'h': [(idx, -coeff) for idx, coeff in B01['h']]
    }
    B_M4 = combine_blocks(B10, B_M4_neg)
    C_M4 = {
        'c00': C00['c00'], # Coeff +1
        'c01': [],
        'c10': [],
        'c11': []
    }
    add_strassen_2x2(A_M4, B_M4, C_M4)
    
    # M5 = (A00+A01)B11 -> C10, C11
    A_M5 = combine_blocks(A00, A01)
    B_M5 = B11
    C_M5 = {
        'c00': [],
        'c01': [],
        'c10': C10['c00'], # Coeff +1
        'c11': C11['c00'] # Coeff -1
    }
    add_strassen_2x2(A_M5, B_M5, C_M5)
    
    # M6 = (A10-A11)(B00+B01) -> C11
    A_M6_neg = {
        'a': [(idx, -coeff) for idx, coeff in A11['a']],
        'b': [(idx, -coeff) for idx, coeff in A11['b']],
        'c': [(idx, -coeff) for idx, coeff in A11['c']],
        'd': [(idx, -coeff) for idx, coeff in A11['d']]
    }
    A_M6 = combine_blocks(A10, A_M6_neg)
    B_M6 = combine_blocks(B00, B01)
    C_M6 = {
        'c00': [],
        'c01': [],
        'c10': [],
        'c11': C11['c00'] # Coeff +1
    }
    add_strassen_2x2(A_M6, B_M6, C_M6)
    
    # M7 = (A01-A10)(B11+B10) -> C00, C01
    A_M7_neg = {
        'a': [(idx, -coeff) for idx, coeff in A10['a']],
        'b': [(idx, -coeff) for idx, coeff in A10['b']],
        'c': [(idx, -coeff) for idx, coeff in A10['c']],
        'd': [(idx, -coeff) for idx, coeff in A10['d']]
    }
    A_M7 = combine_blocks(A01, A_M7_neg)
    B_M7 = combine_blocks(B11, B10)
    C_M7 = {
        'c00': C00['c00'], # Coeff -1
        'c01': C01['c00'], # Coeff +1
        'c10': [],
        'c11': []
    }
    add_strassen_2x2(A_M7, B_M7, C_M7)
    
    # Block 2: Cols 4-7 of B, Cols 4-7 of C
    # B2 Blocks (cols 4-7)
    B20 = {'e': [(36,1)], 'f': [(37,1)], 'g': [(45,1)], 'h': [(46,1)]} # 4*9+4=40? No.
    # B index: j*9 + k.
    # j=0, k=4 -> 4. j=0, k=5 -> 5. j=1, k=4 -> 13. j=1, k=5 -> 14.
    # j=2, k=4 -> 22. j=2, k=5 -> 23. j=3, k=4 -> 31. j=3, k=5 -> 32.
    
    # Let's re-calculate B2 indices.
    # B is 4x9.
    # Block B20 (rows 0-1, cols 4-5):
    # (0,4)=4, (0,5)=5, (1,4)=13, (1,5)=14
    # Block B21 (rows 0-1, cols 6-7):
    # (0,6)=6, (0,7)=7, (1,6)=15, (1,7)=16
    # Block B22 (rows 2-3, cols 4-5):
    # (2,4)=22, (2,5)=23, (3,4)=31, (3,5)=32
    # Block B23 (rows 2-3, cols 6-7):
    # (2,6)=24, (2,7)=25, (3,6)=33, (3,7)=34
    
    B2_00 = {'e': [(4,1)], 'f': [(5,1)], 'g': [(13,1)], 'h': [(14,1)]}
    B2_01 = {'e': [(6,1)], 'f': [(7,1)], 'g': [(15,1)], 'h': [(16,1)]}
    B2_10 = {'e': [(22,1)], 'f': [(23,1)], 'g': [(31,1)], 'h': [(32,1)]}
    B2_11 = {'e': [(24,1)], 'f': [(25,1)], 'g': [(33,1)], 'h': [(34,1)]}
    
    # C2 Blocks (cols 4-7)
    # C index: k*4 + i.
    # k=4..7, i=0..3.
    # C20 (cols 4-5, rows 0-1):
    # (4,0)=16, (4,1)=17, (5,0)=20, (5,1)=21
    # C21 (cols 4-5, rows 2-3):
    # (4,2)=18, (4,3)=19, (5,2)=22, (5,3)=23
    # C22 (cols 6-7, rows 0-1):
    # (6,0)=24, (6,1)=25, (7,0)=28, (7,1)=29
    # C23 (cols 6-7, rows 2-3):
    # (6,2)=26, (6,3)=27, (7,2)=30, (7,3)=31
    
    C2_00 = {'c00': [(16,1)], 'c01': [(17,1)], 'c10': [(20,1)], 'c11': [(21,1)]}
    C2_01 = {'c00': [(18,1)], 'c01': [(19,1)], 'c10': [(22,1)], 'c11': [(23,1)]}
    C2_10 = {'c00': [(24,1)], 'c01': [(25,1)], 'c10': [(28,1)], 'c11': [(29,1)]}
    C2_11 = {'c00': [(26,1)], 'c01': [(27,1)], 'c10': [(30,1)], 'c11': [(31,1)]}
    
    # Repeat Strassen for Block 2
    A_M1_2 = combine_blocks(A00, A11)
    B_M1_2 = combine_blocks(B2_00, B2_11)
    C_M1_2 = {'c00': C2_00['c00'], 'c01': [], 'c10': C2_01['c00'], 'c11': []}
    add_strassen_2x2(A_M1_2, B_M1_2, C_M1_2)
    
    A_M2_2 = combine_blocks(A10, A11)
    B_M2_2 = B2_00
    C_M2_2 = {'c00': C2_00['c00'], 'c01': [], 'c10': [], 'c11': C2_11['c00']}
    add_strassen_2x2(A_M2_2, B_M2_2, C_M2_2)
    
    A_M3_2 = A00
    B_M3_2_neg = {k: [(i, -c) for i, c in B2_11.items()] for k in B2_11} # Wrong logic
    # Fix negation
    B_M3_2_neg = {
        'e': [(idx, -coeff) for idx, coeff in B2_11['e']],
        'f': [(idx, -coeff) for idx, coeff in B2_11['f']],
        'g': [(idx, -coeff) for idx, coeff in B2_11['g']],
        'h': [(idx, -coeff) for idx, coeff in B2_11['h']]
    }
    B_M3_2 = combine_blocks(B2_01, B_M3_2_neg)
    C_M3_2 = {'c00': [], 'c01': C2_00['c01'], 'c10': [], 'c11': []}
    add_strassen_2x2(A_M3_2, B_M3_2, C_M3_2)
    
    A_M4_2 = A11
    B_M4_2_neg = {
        'e': [(idx, -coeff) for idx, coeff in B2_01['e']],
        'f': [(idx, -coeff) for idx, coeff in B2_01['f']],
        'g': [(idx, -coeff) for idx, coeff in B2_01['g']],
        'h': [(idx, -coeff) for idx, coeff in B2_01['h']]
    }
    B_M4_2 = combine_blocks(B2_10, B_M4_2_neg)
    C_M4_2 = {'c00': C2_00['c00'], 'c01': [], 'c10': [], 'c11': []}
    add_strassen_2x2(A_M4_2, B_M4_2, C_M4_2)
    
    A_M5_2 = combine_blocks(A00, A01)
    B_M5_2 = B2_11
    C_M5_2 = {'c00': [], 'c01': [], 'c10': C2_01['c00'], 'c11': C2_11['c00']}
    add_strassen_2x2(A_M5_2, B_M5_2, C_M5_2)
    
    A_M6_2_neg = {
        'a': [(idx, -coeff) for idx, coeff in A11['a']],
        'b': [(idx, -coeff) for idx, coeff in A11['b']],
        'c': [(idx, -coeff) for idx, coeff in A11['c']],
        'd': [(idx, -coeff) for idx, coeff in A11['d']]
    }
    A_M6_2 = combine_blocks(A10, A_M6_2_neg)
    B_M6_2 = combine_blocks(B2_00, B2_01)
    C_M6_2 = {'c00': [], 'c01': [], 'c10': [], 'c11': C2_11['c00']}
    add_strassen_2x2(A_M6_2, B_M6_2, C_M6_2)
    
    A_M7_2_neg = {
        'a': [(idx, -coeff) for idx, coeff in A10['a']],
        'b': [(idx, -coeff) for idx, coeff in A10['b']],
        'c': [(idx, -coeff) for idx, coeff in A10['c']],
        'd': [(idx, -coeff) for idx, coeff in A10['d']]
    }
    A_M7_2 = combine_blocks(A01, A_M7_2_neg)
    B_M7_2 = combine_blocks(B2_11, B2_10)
    C_M7_2 = {'c00': C2_00['c00'], 'c01': C2_00['c01'], 'c10': [], 'c11': []}
    add_strassen_2x2(A_M7_2, B_M7_2, C_M7_2)
    
    # Block 3: Col 8 of B, Col 8 of C
    # Schoolbook 4x4x1
    # B col 8: indices 8, 17, 26, 35
    # C col 8: indices 32, 33, 34, 35
    
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            add_rank1(u_idx, v_idx, w_idx, 1)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
