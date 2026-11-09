"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Using block-Strassen decomposition
    
    # Helper: add a 2x2 Strassen product into the global U,V,W
    # U_local: 4 entries (2x2) -> indices in U
    # V_local: 4 entries (2x2) -> indices in V  
    # W_local: 4 entries (2x2) -> indices in W
    # The product: C_local[i][j] = sum_k U_global[...]*V_global[...]*W_global[...]
    
    # We'll accumulate contributions
    # Start with empty lists, then build arrays
    
    # Each Strassen 2x2x2 product uses 7 terms
    # We'll track which global column each term goes into
    
    rank = 0
    # Store as lists of (row, col) contributions
    U_rows = []  # list of (row_index, value) per term
    V_rows = []
    W_rows = []
    
    def add_strassen_2x2(U_idx_base, V_idx_base, W_idx_base, sign=1):
        """Add a Strassen 2x2 product. U_idx_base, V_idx_base, W_idx_base are
        the starting indices in the global U,V,W row spaces for the 2x2 blocks.
        Returns the number of columns added."""
        nonlocal rank
        added = 0
        # Strassen 2x2: M1=(a+d)(e+h), M2=(b+c)d, M3=a(e-f), M4=(a+d)(g+h),
        # M5=(c+d)h, M6=(a-b)(e+f), M7=(c-b)(g+h)
        # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
        
        # Each M is a product of two linear combinations
        # We'll add 7 terms
        
        # For each Strassen multiplier, we need to figure out the U, V, W contributions
        # U has 2x2 = 4 entries for the left matrix block
        # V has 2x2 = 4 entries for the right matrix block  
        # W has 2x2 = 4 entries for the output block
        
        # U indices: U_idx_base + [0,1,2,3] corresponding to [[0,0],[0,1],[1,0],[1,1]]
        # V indices: V_idx_base + [0,1,2,3] corresponding to [[0,0],[0,1],[1,0],[1,1]]
        # W indices: W_idx_base + [0,1,2,3] corresponding to [[0,0],[0,1],[1,0],[1,1]]
        
        # M1 = (a+d)(e+h): U gets a=1,d=1 (positions 0,2), V gets e=1,h=1 (positions 0,2)
        # M2 = (b+c)d: U gets b=1,c=1 (positions 1,3), V gets d=1 (position 1... wait)
        
        # Let me be more careful. For 2x2 matrices:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        
        # U maps to A: U[0]=a, U[1]=b, U[2]=c, U[3]=d
        # V maps to B: V[0]=e, V[1]=f, V[2]=g, V[3]=h
        # W maps to C: W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11
        
        # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; contributes to C00, C01, C10, C11
        # Actually in Strassen, each M contributes to specific C entries with coefficients.
        
        # Let me just add 7 terms directly with the known Strassen structure.
        
        # I'll represent each term as (U_contrib, V_contrib, W_contrib)
        # where each is a dict mapping local index to coefficient
        
        terms = []
        # M1 = (a+d)(e+h) -> contributes: C00 += 1, C01 += 1, C10 += 1, C11 += 1? No.
        # Standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (b+c)d  
        # M3 = a(e-f)
        # M4 = (a+d)(g+h)
        # M5 = (c+d)h
        # M6 = (a-b)(e+f)
        # M7 = (c-b)(g+h)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: U={0:1, 2:1}, V={0:1, 3:1}, W={0:1, 1:1, 2:1, 3:1}? No, W should only have entries where it contributes.
        # Actually W is the output tensor. For M1, it contributes to all of C with coefficient 1 in C00, and... 
        # Let me just say: for each M, the W contribution is determined by which C entries it affects and with what coefficient.
        
        # M1 contributes: C00 with coeff 1, C01 with coeff 0? No.
        # C00 = M1 + M4 - M5 + M7: so M1 contributes 1 to C00
        # C01 = M3 + M5: M1 contributes 0 to C01
        # C10 = M2 + M4: M1 contributes 0 to C10  
        # C11 = M1 - M2 + M3 + M6: M1 contributes 1 to C11
        
        # So W for M1: {0: 1, 3: 1} (C00 and C11)
        
        # Let me define all 7:
        # M1: U={0:1,2:1}, V={0:1,3:1}, W={0:1, 3:1}
        # M2: U={1:1,3:1}, V={3:1}, W={2:1, 3:-1}  (C10 += 1, C11 -= 1)
        # M3: U={0:1}, V={0:1,2:-1}, W={1:1, 3:1}  (C01 += 1, C11 += 1)
        # M4: U={0:1,2:1}, V={2:1,3:1}, W={0:1, 2:1}  (C00 += 1, C10 += 1)
        # M5: U={2:1,3:1}, V={3:1}, W={0:-1, 1:1}  (C00 -= 1, C01 += 1)
        # M6: U={0:1,1:-1}, V={0:1,1:1}, W={3:1}  (C11 += 1)
        # M7: U={2:1,1:-1}, V={2:1,3:1}, W={0:1}  (C00 += 1)
        
        for (U_d, V_d, W_d) in [
            ({0:1,2:1}, {0:1,3:1}, {0:1, 3:1}),       # M1
            ({1:1,3:1}, {3:1}, {2:1, 3:-1}),           # M2
            ({0:1}, {0:1,2:-1}, {1:1, 3:1}),           # M3
            ({0:1,2:1}, {2:1,3:1}, {0:1, 2:1}),        # M4
            ({2:1,3:1}, {3:1}, {0:-1, 1:1}),           # M5
            ({0:1,1:-1}, {0:1,1:1}, {3:1}),            # M6
            ({2:1,1:-1}, {2:1,3:1}, {0:1}),            # M7
        ]:
            col = rank
            for idx, val in U_d.items():
                U_rows.append((U_idx_base + idx, val))
            for idx, val in V_d.items():
                V_rows.append((V_idx_base + idx, val))
            for idx, val in W_d.items():
                W_rows.append((W_idx_base + idx, val))
            rank += 1
            added += 1
        return added
    
    # Now decompose the 4x4 x 4x6 product
    # Split into left 4x4 and right 4x2 parts
    
    # Left part: A(4x4) * B_left(4x4) -> C_left(4x4)
    # Split A into 2x2 blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
    #                        A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # Split B_left similarly: B00, B01, B10, B11
    # C_left[0:2,0:2] = A00*B00 + A01*B10
    # C_left[0:2,2:3] = A00*B01 + A01*B11  (wait, cols 2-3)
    # C_left[2:4,0:2] = A10*B00 + A11*B10
    # C_left[2:4,2:4] = A10*B01 + A11*B11
    
    # Each of these is a 2x2 x 2x2 -> 2x2 product, rank 7 each
    # 4 such products: rank 28
    
    # For U: A is 4x4, indexed as A[i*m+j] = A[i*4+j]
    # A00: i in 0-1, j in 0-1 -> U indices 0,1,4,5
    # A01: i in 0-1, j in 2-3 -> U indices 2,3,6,7
    # A10: i in 2-3, j in 0-1 -> U indices 8,9,12,13
    # A11: i in 2-3, j in 2-3 -> U indices 10,11,14,15
    
    # For V: B is 4x6, indexed as B[k*p+l] = B[k*6+l]
    # B_left is cols 0-3 of B
    # B00: k in 0-1, l in 0-1 -> V indices 0,1,6,7
    # B01: k in 0-1, l in 2-3 -> V indices 2,3,8,9
    # B10: k in 2-3, l in 0-1 -> V indices 12,13,18,19
    # B11: k in 2-3, l in 2-3 -> V indices 14,15,20,21
    
    # For W: C is 4x4 (left part), indexed as C[k*n+i] = C[k*4+i] per the convention W[k*n+i,r]
    # Wait, the convention is W[k*n+i,r] for the output. The output is 4x4 for left part.
    # C_left[i,j] where i is row (0-3), j is col (0-3). W index = j*4 + i? Or i*4+j?
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W index = k*n + i where k is column index of C, i is row index of C.
    # For left part, C is 4x4, so W indices 0-15.
    # C[0,0] -> W[0*4+0]=0, C[0,1] -> W[1*4+0]=4, C[1,0] -> W[0*4+1]=1, etc.
    # C_top_left (rows 0-1, cols 0-1): W indices 0,4,1,5
    # C_top_right (rows 0-1, cols 2-3): W indices 8,12,9,13
    # C_bot_left (rows 2-3, cols 0-1): W indices 2,6,3,7
    # C_bot_right (rows 2-3, cols 2-3): W indices 10,14,11,15
    
    # Right part: A(4x4) * B_right(4x2) -> C_right(4x2)
    # B_right is cols 4-5 of B
    # Split similarly:
    # B_right_top (rows 0-1, cols 4-5): V indices 4,5,10,11
    # B_right_bot (rows 2-3, cols 4-5): V indices 16,17,22,23
    # C_right_top (rows 0-1, cols 0-1): W indices 16,20,17,21
    # C_right_bot (rows 2-3, cols 0-1): W indices 18,22,19,23
    
    # C_right[0:2,:] = A00*B_right_top + A01*B_right_bot
    # C_right[2:4,:] = A10*B_right_top + A11*B_right_bot
    # 4 products of 2x2 x 2x2 -> 2x2, rank 28
    
    # Total rank: 28 + 28 = 56? Wait, that's for the block decomposition.
    # But we're not doing full Strassen on the 4x4x4, we're doing 4 separate 2x2 products.
    # That gives 4*7 = 28 for left, 4*7 = 28 for right, total 56.
    
    # Hmm, but the standard Strassen for 4x4x4 gives rank 49, not 56.
    # The 4x4x4 can be done with 7 Strassen steps at the 4x4 level? No.
    # Actually, 4x4x4: split into 2x2 blocks, you get 8 products of 2x2x2x2... 
    # No, it's: C = A*B where A,B,C are 4x4. Split into 2x2 blocks.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each sum has 2 products, so 8 products total, each rank 7, total 56.
    # But Strassen says rank 49 for 4x4x4. How?
    # Ah, I see - you can apply Strassen at the 4x4 level treating the 2x2 blocks as "scalars".
    # So the 4x4x4 becomes a 2x2x2 product of 2x2 matrices, which has rank 7,
    # and each "scalar" multiplication is a 2x2x2x2 product of rank 7.
    # Total rank: 7 * 7 = 49.
    
    # So I need to do the Strassen at the block level, where each block product
    # is itself a 2x2x2x2 tensor product.
    
    # This is getting complex. Let me just use the 56 rank approach for now,
    # or find a cleaner way.
    
    # Actually, let me just implement the direct 4x4x4 with 8 block products (rank 56)
    # and 4x4x2 with 4 block products (rank 28), total 84.
    # That's worse than 77.
    
    # Let me just go with the simpler approach: do each 2x2 subproduct separately.
    # For 4x4x4: 8 products * 7 = 56
    # For 4x4x2: 4 products * 7 = 28
    # Total: 84. Still better than 96.
    
    # Let me implement this.
    
    # Reset
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # U indices for A (4x4): A[i*4+j]
    # A00: rows 0-1, cols 0-1: indices 0,1,4,5
    # A01: rows 0-1, cols 2-3: indices 2,3,6,7
    # A10: rows 2-3, cols 0-1: indices 8,9,12,13
    # A11: rows 2-3, cols 2-3: indices 10,11,14,15
    
    # V indices for B (4x6): B[k*6+l]
    # Left part (cols 0-3):
    # B00: rows 0-1, cols 0-1: indices 0,1,6,7
    # B01: rows 0-1, cols 2-3: indices 2,3,8,9
    # B10: rows 2-3, cols 0-1: indices 12,13,18,19
    # B11: rows 2-3, cols 2-3: indices 14,15,20,21
    # Right part (cols 4-5):
    # Br_top: rows 0-1, cols 4-5: indices 4,5,10,11
    # Br_bot: rows 2-3, cols 4-5: indices 16,17,22,23
    
    # W indices for C (4x6): C[k*4+i] where k=col, i=row
    # Left part:
    # C00: rows 0-1, cols 0-1: W[0*4+0]=0, W[1*4+0]=4, W[0*4+1]=1, W[1*4+1]=5 -> {0:0, 4:0, 1:1, 5:1}
    # C01: rows 0-1, cols 2-3: W[2*4+0]=8, W[3*4+0]=12, W[2*4+1]=9, W[3*4+1]=13 -> {8:0, 12:0, 9:1, 13:1}
    # C10: rows 2-3, cols 0-1: W[0*4+2]=2, W[1*4+2]=6, W[0*4+3]=3, W[1*4+3]=7 -> {2:2, 6:2, 3:3, 7:3}
    # C11: rows 2-3, cols 2-3: W[2*4+2]=10, W[3*4+2]=14, W[2*4+3]=11, W[3*4+3]=15 -> {10:2, 14:2, 11:3, 15:3}
    # Right part:
    # Cr_top: rows 0-1, cols 0-1: W[4*4+0]=16, W[5*4+0]=20, W[4*4+1]=17, W[5*4+1]=21 -> {16:0, 20:0, 17:1, 21:1}
    # Cr_bot: rows 2-3, cols 0-1: W[4*4+2]=18, W[5*4+2]=22, W[4*4+3]=19, W[5*4+3]=23 -> {18:2, 22:2, 19:3, 23:3}
    
    # For each 2x2 product, I need to map the local 2x2 indices to global.
    # Local 2x2: [[0,1],[2,3]] mapping to global indices.
    
    # Define the block products:
    # (U_block_indices, V_block_indices, W_block_indices)
    
    block_products = []
    
    # Left part:
    # C00 = A00*B00 + A01*B10
    block_products.append(([0,1,4,5], [0,1,6,7], [0,4,1,5]))     # A00*B00 -> C00
    block_products.append(([2,3,6,7], [12,13,18,19], [0,4,1,5]))  # A01*B10 -> C00
    
    # C01 = A00*B01 + A01*B11
    block_products.append(([0,1,4,5], [2,3,8,9], [8,12,9,13]))     # A00*B01 -> C01
    block_products.append(([2,3,6,7], [14,15,20,21], [8,12,9,13])) # A01*B11 -> C01
    
    # C10 = A10*B00 + A11*B10
    block_products.append(([8,9,12,13], [0,1,6,7], [2,6,3,7]))     # A10*B00 -> C10
    block_products.append(([10,11,14,15], [12,13,18,19], [2,6,3,7])) # A11*B10 -> C10
    
    # C11 = A10*B01 + A11*B11
    block_products.append(([8,9,12,13], [2,3,8,9], [10,14,11,15]))     # A10*B01 -> C11
    block_products.append(([10,11,14,15], [14,15,20,21], [10,14,11,15])) # A11*B11 -> C11
    
    # Right part:
    # Cr_top = A00*Br_top + A01*Br_bot
    block_products.append(([0,1,4,5], [4,5,10,11], [16,20,17,21]))     # A00*Br_top -> Cr_top
    block_products.append(([2,3,6,7], [16,17,22,23], [16,20,17,21]))   # A01*Br_bot -> Cr_top
    
    # Cr_bot = A10*Br_top + A11*Br_bot
    block_products.append(([8,9,12,13], [4,5,10,11], [18,22,19,23]))     # A10*Br_top -> Cr_bot
    block_products.append(([10,11,14,15], [16,17,22,23], [18,22,19,23])) # A11*Br_bot -> Cr_bot
    
    # Now add each block product using Strassen
    for U_idx, V_idx, W_idx in block_products:
        # U_idx is [u00, u01, u10, u11] global indices
        # Local index 0->U_idx[0], 1->U_idx[1], 2->U_idx[2], 3->U_idx[3]
        # Same for V and W
        
        terms = [
            ({0:1,2:1}, {0:1,3:1}, {0:1, 3:1}),       # M1
            ({1:1,3:1}, {3:1}, {2:1, 3:-1}),           # M2
            ({0:1}, {0:1,2:-1}, {1:1, 3:1}),           # M3
            ({0:1,2:1}, {2:1,3:1}, {0:1, 2:1}),        # M4
            ({2:1,3:1}, {3:1}, {0:-1, 1:1}),           # M5
            ({0:1,1:-1}, {0:1,1:1}, {3:1}),            # M6
            ({2:1,1:-1}, {2:1,3:1}, {0:1}),            # M7
        ]
        
        for (U_d, V_d, W_d) in terms:
            col = rank
            for local_idx, val in U_d.items():
                U_rows.append((U_idx[local_idx], val))
            for local_idx, val in V_d.items():
                V_rows.append((V_idx[local_idx], val))
            for local_idx, val in W_d.items():
                W_rows.append((W_idx[local_idx], val))
            rank += 1
    
    # Build the matrices
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 24 x rank
    W = [[0] * rank for _ in range(p * n)]  # 24 x rank
    
    for (row, val) in U_rows:
        U[row][rank - len([r for r in U_rows[:rank]])]  # This is getting messy
    
    # Let me redo this more carefully.
    # Each term adds one column. I'll track columns explicitly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for U_idx, V_idx, W_idx in block_products:
        terms = [
            ({0:1,2:1}, {0:1,3:1}, {0:1, 3:1}),
            ({1:1,3:1}, {3:1}, {2:1, 3:-1}),
            ({0:1}, {0:1,2:-1}, {1:1, 3:1}),
            ({0:1,2:1}, {2:1,3:1}, {0:1, 2:1}),
            ({2:1,3:1}, {3:1}, {0:-1, 1:1}),
            ({0:1,1:-1}, {0:1,1:1}, {3:1}),
            ({2:1,1:-1}, {2:1,3:1}, {0:1}),
        ]
        for (U_d, V_d, W_d) in terms:
            for local_idx, val in U_d.items():
                U[U_idx[local_idx]][col_idx] = val
            for local_idx, val in V_d.items():
                V[V_idx[local_idx]][col_idx] = val
            for local_idx, val in W_d.items():
                W[W_idx[local_idx]][col_idx] = val
            col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

