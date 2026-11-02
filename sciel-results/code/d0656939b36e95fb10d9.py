"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We'll build rank-1 terms: each term is (u, v, w) where
    # u has length n*m=20, v has length m*p=30, w has length p*n=24
    # The decomposition: sum_k u_k[i*m+j] * v_k[j*p+k] * w_k[k*n+i] = C[i,k]
    
    # Strategy: 
    # 1. 4x4x4 Strassen (rank 49) for C[0:4, 0:4]
    # 2. 4x4x2 schoolbook (rank 32) for C[0:4, 4:6]  
    # 3. 4x1x6 schoolbook (rank 24) for C[4:5, 0:6]
    # Total rank: 49 + 32 + 24 = 105
    
    terms = []  # list of (u_dict, v_dict, w_dict) where dicts map index->coeff
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: list of (idx, coeff), etc."""
        terms.append((u_entries, v_entries, w_entries))
    
    # === Part 1: 4x4x4 Strassen (rank 49) ===
    # A sub: rows 0-3, cols 0-3 of A (4x4)
    # B sub: rows 0-3, cols 0-3 of B (4x4)
    # C sub: rows 0-3, cols 0-3 of C (4x4)
    
    # Recursive Strassen: 4x4x4 -> 7 terms of 2x2x2
    # Each 2x2x2 has rank 7, so 7*7=49 total
    
    def strassen_2x2_terms():
        """Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
        Each term is (u, v, w) where u,v,w are lists of 4 entries.
        u[i*2+j] = coeff for A[i,j]
        v[i*2+j] = coeff for B[i,j]  
        w[i*2+j] = coeff for C[i,j]
        """
        # Standard Strassen 2x2 decomposition
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mi is a scalar product of combined entries.
        # But in tensor decomposition, each rank-1 term corresponds to one "multiplication"
        # where the u, v, w vectors have entries that are sums.
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # u: A00=1, A11=1 -> [1,0,0,1]
        # v: B00=1, B11=1 -> [1,0,0,1]
        # w: C00=1, C11=1 -> [1,0,0,1]
        
        # Term 2: M2 = (A10+A11)B00
        # u: A10=1, A11=1 -> [0,0,1,1]
        # v: B00=1 -> [1,0,0,0]
        # w: C10=1, C11=-1 -> [0,0,1,-1]
        
        # Term 3: M3 = A00(B01-B11)
        # u: A00=1 -> [1,0,0,0]
        # v: B01=1, B11=-1 -> [0,1,0,-1]
        # w: C01=1, C11=1 -> [0,1,0,1]
        
        # Term 4: M4 = A11(B10-B00)
        # u: A11=1 -> [0,0,0,1]
        # v: B10=1, B00=-1 -> [0,-1,1,0]
        # w: C00=1, C10=1 -> [1,0,1,0]
        
        # Term 5: M5 = (A00+A01)B11
        # u: A00=1, A01=1 -> [1,1,0,0]
        # v: B11=1 -> [0,0,0,1]
        # w: C00=-1, C01=1 -> [-1,1,0,0]
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # u: A10=1, A00=-1 -> [-1,0,1,0]
        # v: B00=1, B01=1 -> [1,1,0,0]
        # w: C11=1 -> [0,0,0,1]
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # u: A01=1, A11=-1 -> [0,1,0,-1]
        # v: B10=1, B11=1 -> [0,0,1,1]
        # w: C00=1, C11=1 -> [1,0,0,1]
        
        return [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
            ([0,0,0,1], [0,-1,1,0], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,1])
        ]
    
    s2 = strassen_2x2_terms()
    
    # For 4x4x4 Strassen, we split into 2x2 blocks:
    # A blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
    # B blocks: B00=B[0:2,0:2], B01=B[0:2,2:4], B10=B[2:4,0:2], B11=B[2:4,2:4]
    # C blocks: C00=C[0:2,0:2], C01=C[0:2,2:4], C10=C[2:4,0:2], C11=C[2:4,2:4]
    
    # Top-level Strassen 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2 matrix product. We decompose each using s2 terms.
    # For each top-level term, we define which A-blocks and B-blocks are involved
    # with what coefficients, then for each s2 sub-term, we map to global indices.
    
    # Define block mappings:
    # A block (br, bc) -> global A indices: rows br*2 to br*2+1, cols bc*2 to bc*2+1
    # So A[br*2+r, bc*2+c] for r,c in 0,1
    # Global A index: (br*2+r) * 4 + (bc*2+c)  [but A is 4x5, so use 5 for stride]
    # Actually A is n x m = 4 x 5, so A[i,j] has index i*5+j
    # B is m x p = 5 x 6, so B[i,j] has index i*6+j
    # C is n x p = 4 x 6, so C[i,j] has index i*6+j
    # W is p*n = 6*4 = 24, W[k*n+i] = W[k*4+i]
    
    def a_global(br, bc, r, c):
        """Global A index for block (br,bc), local (r,c). A is 4x5."""
        return (br*2 + r) * 5 + (bc*2 + c)
    
    def b_global(br, bc, r, c):
        """Global B index for block (br,bc), local (r,c). B is 5x6."""
        return (br*2 + r) * 6 + (bc*2 + c)
    
    def c_global(br, bc, r, c):
        """Global C index for block (br,bc), local (r,c). C is 4x6."""
        return (br*2 + r) * 6 + (bc*2 + c)
    
    # For each top-level Strassen term, define:
    # a_blocks: list of (br, bc, coeff)
    # b_blocks: list of (br, bc, coeff)
    # c_blocks: list of (br, bc, coeff)
    
    top_terms = [
        # M1 = (A00+A11)(B00+B11)
        {
            'a': [(0,0,1), (1,1,1)],
            'b': [(0,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,1)]
        },
        # M2 = (A10+A11)B00
        {
            'a': [(1,0,1), (1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1), (1,1,-1)]
        },
        # M3 = A00(B01-B11)
        {
            'a': [(0,0,1)],
            'b': [(0,1,1), (1,1,-1)],
            'c': [(0,1,1), (1,1,1)]
        },
        # M4 = A11(B10-B00)
        {
            'a': [(1,1,1)],
            'b': [(1,0,1), (0,0,-1)],
            'c': [(0,0,1), (1,0,1)]
        },
        # M5 = (A00+A01)B11
        {
            'a': [(0,0,1), (0,1,1)],
            'b': [(1,1,1)],
            'c': [(0,0,-1), (0,1,1)]
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'a': [(1,0,1), (0,0,-1)],
            'b': [(0,0,1), (0,1,1)],
            'c': [(1,1,1)]
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'a': [(0,1,1), (1,1,-1)],
            'b': [(1,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,1)]
        }
    ]
    
    for top in top_terms:
        a_blocks = top['a']
        b_blocks = top['b']
        c_blocks = top['c']
        
        # For each of the 7 s2 sub-terms, build global u, v, w
        for s2_u, s2_v, s2_w in s2:
            u_entries = []
            v_entries = []
            w_entries = []
            
            # s2_u = [u00, u01, u10, u11] corresponds to local A entries (0,0), (0,1), (1,0), (1,1)
            # We need to expand these through the block definitions.
            # For local A entry (lr, lc) with coeff s2_u[lr*2+lc]:
            #   This local entry is a sum over a_blocks: sum of coeff_a * A_block(br,bc)[lr,lc]
            #   Wait, no. The local A matrix for this top-level term is:
            #   X[lr,lc] = sum over (br,bc,ca) in a_blocks: ca * A_global[br*2+lr, bc*2+lc]
            #   Then the s2 term picks X[lr,lc] with coefficient s2_u[lr*2+lc]
            #   So global contribution: for each (lr,lc) with s2_u[lr*2+lc] != 0,
            #     for each (br,bc,ca) in a_blocks:
            #       global_a_idx = a_global(br, bc, lr, lc)
            #       coeff = s2_u[lr*2+lc] * ca
            #       add to u_entries
            
            for lr in range(2):
                for lc in range(2):
                    su = s2_u[lr*2+lc]
                    if su == 0:
                        continue
                    for br, bc, ca in a_blocks:
                        gidx = a_global(br, bc, lr, lc)
                        coeff = su * ca
                        if coeff != 0:
                            u_entries.append((gidx, coeff))
            
            # Similarly for v
            for lr in range(2):
                for lc in range(2):
                    sv = s2_v[lr*2+lc]
                    if sv == 0:
                        continue
                    for br, bc, cb in b_blocks:
                        gidx = b_global(br, bc, lr, lc)
                        coeff = sv * cb
                        if coeff != 0:
                            v_entries.append((gidx, coeff))
            
            # Similarly for w
            for lr in range(2):
                for lc in range(2):
                    sw = s2_w[lr*2+lc]
                    if sw == 0:
                        continue
                    for br, bc, cc in c_blocks:
                        gidx = c_global(br, bc, lr, lc)
                        coeff = sw * cc
                        if coeff != 0:
                            w_entries.append((gidx, coeff))
            
            add_term(u_entries, v_entries, w_entries)
    
    # === Part 2: 4x4x2 schoolbook (rank 32) ===
    # C[0:4, 4:6] = A[0:4, 0:4] * B[0:4, 4:6]
    # A sub: rows 0-3, cols 0-3 (4x4)
    # B sub: rows 0-3, cols 4-5 (4x2)
    # C sub: rows 0-3, cols 4-5 (4x2)
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u_entries = [(i * 5 + j, 1)]
                v_entries = [(j * 6 + (4+k), 1)]
                w_entries = [((4+k) * 4 + i, 1)]
                add_term(u_entries, v_entries, w_entries)
    
    # === Part 3: 4x1x6 schoolbook (rank 24) ===
    # C[4:5, 0:6] = A[4:5, 0:4] * B[0:4, 0:6]
    # A sub: row 4, cols 0-3 (1x4)
    # B sub: rows 0-3, cols 0-5 (4x6)
    # C sub: row 4, cols 0-5 (1x6)
    
    for i in range(4):
        for j in range(4):
            for k in range(6):
                u_entries = [(4 * 5 + j, 1)]  # A[4, j]
                v_entries = [(j * 6 + k, 1)]  # B[j, k]
                w_entries = [(k * 4 + 4, 1)]  # C[4, k]
                add_term(u_entries, v_entries, w_entries)
    
    # Now assemble U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, coeff in u_entries:
            U[idx][col] += coeff
        for idx, coeff in v_entries:
            V[idx][col] += coeff
        for idx, coeff in w_entries:
            W[idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
