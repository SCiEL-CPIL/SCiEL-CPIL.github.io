DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Total rank
    rank_part1 = 49
    rank_part2 = 48
    rank = rank_part1 + rank_part2
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x rank
    V = [[0] * rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * rank for _ in range(p * n)]  # 28 x rank
    
    col = 0
    
    # --- Part 1: 4x4 x 4x4 (columns 0..3 of B) ---
    # Strassen for 4x4 using 2x2 blocks
    
    # Define the 7 Strassen linear combinations for 4x4 matrices
    # Each is a tuple of (A_coeffs, B_coeffs)
    # A_coeffs: dict mapping (block_row, block_col) to coeff
    # B_coeffs: dict mapping (block_row, block_col) to coeff
    # Blocks are 2x2.
    # A blocks: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B blocks: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    
    strassen_4x4 = [
        # M1 = (A11+A22)(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A21+A22)B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        # M3 = A11(B12-B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # M4 = A22(B21-B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        # M5 = (A11+A12)B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M6 = (A21-A11)(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        # M7 = (A12-A22)(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
    ]
    
    # Define the 7 rank-1 terms for 2x2 matrix multiplication (Strassen)
    # Each term is (U_2x2, V_2x2, W_2x2)
    # U, V, W are 2x2 lists
    def make_2x2(val):
        return [[val]*2 for _ in range(2)]
        
    zero_2x2 = [[0]*2 for _ in range(2)]
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # U: 1 at (0,0), (1,1)
    # V: 1 at (0,0), (1,1)
    # W: 1 at (0,0), (1,1)
    t1_U = [[1,0],[0,1]]
    t1_V = [[1,0],[0,1]]
    t1_W = [[1,0],[0,1]]
    
    # Term 2: (A10+A11)B00 -> C10, C11
    # U: 1 at (1,0), (1,1)
    # V: 1 at (0,0)
    # W: 1 at (0,1), (1,1)
    t2_U = [[0,0],[1,1]]
    t2_V = [[1,0],[0,0]]
    t2_W = [[0,1],[0,1]]
    
    # Term 3: A00(B01-B11) -> C01, C11
    # U: 1 at (0,0)
    # V: 1 at (0,1), -1 at (1,1)
    # W: 1 at (1,0), 1 at (1,1)
    t3_U = [[1,0],[0,0]]
    t3_V = [[0,1],[0,-1]]
    t3_W = [[0,1],[0,1]]
    
    # Term 4: A11(B10-B00) -> C10, C11
    # U: 1 at (1,1)
    # V: 1 at (1,0), -1 at (0,0)
    # W: 1 at (0,1), 1 at (1,1)
    t4_U = [[0,0],[0,1]]
    t4_V = [[-1,0],[1,0]]
    t4_W = [[0,1],[0,1]]
    
    # Term 5: (A00+A01)B11 -> C01, C11
    # U: 1 at (0,0), (0,1)
    # V: 1 at (1,1)
    # W: 1 at (1,0), 1 at (1,1)
    t5_U = [[1,1],[0,0]]
    t5_V = [[0,0],[0,1]]
    t5_W = [[0,1],[0,1]]
    
    # Term 6: (A10-A00)(B00+B01) -> C10, C11
    # U: 1 at (1,0), -1 at (0,0)
    # V: 1 at (0,0), (0,1)
    # W: 1 at (0,1), 1 at (1,1)
    t6_U = [[-1,0],[1,0]]
    t6_V = [[1,1],[0,0]]
    t6_W = [[0,1],[0,1]]
    
    # Term 7: (A01-A11)(B10+B11) -> C01, C11
    # U: 1 at (0,1), -1 at (1,1)
    # V: 1 at (1,0), (1,1)
    # W: 1 at (1,0), 1 at (1,1)
    t7_U = [[0,1],[-1,0]]
    t7_V = [[0,0],[1,1]]
    t7_W = [[0,1],[0,1]]
    
    strassen_2x2_terms = [
        (t1_U, t1_V, t1_W),
        (t2_U, t2_V, t2_W),
        (t3_U, t3_V, t3_W),
        (t4_U, t4_V, t4_W),
        (t5_U, t5_V, t5_W),
        (t6_U, t6_V, t6_W),
        (t7_U, t7_V, t7_W),
    ]
    
    # Helper to add a 2x2 multiplication term to global U, V, W
    # a_coeffs: dict (br, bc) -> coeff for A blocks
    # b_coeffs: dict (br, bc) -> coeff for B blocks
    # term: (U_2x2, V_2x2, W_2x2)
    # k_offset: offset for B columns (0 for part 1)
    def add_strassen_4x4_term(a_coeffs, b_coeffs, term, k_offset=0):
        nonlocal col
        U_2x2, V_2x2, W_2x2 = term
        
        # Expand U
        for br in range(2):
            for bc in range(2):
                coeff_a = a_coeffs.get((br, bc), 0)
                if coeff_a == 0: continue
                for u in range(2):
                    for v in range(2):
                        val = coeff_a * U_2x2[u][v]
                        if val == 0: continue
                        global_i = br * 2 + u
                        global_j = bc * 2 + v
                        U[global_i * m + global_j][col] += val
                        
        # Expand V
        for br in range(2):
            for bc in range(2):
                coeff_b = b_coeffs.get((br, bc), 0)
                if coeff_b == 0: continue
                for v in range(2): # v is row of B block, which is col of A block
                    for x in range(2): # x is col of B block
                        val = coeff_b * V_2x2[v][x]
                        if val == 0: continue
                        global_j = br * 2 + v
                        global_k = k_offset + bc * 2 + x
                        V[global_j * p + global_k][col] += val
                        
        # Expand W
        # W index is k*n + i
        # W_2x2[x][u] corresponds to C[x, u]
        # Global C index: (k_offset + bc*2 + x) * n + (br*2 + u)
        # Wait, the W_2x2 term is for the 2x2 result block.
        # The result block is at (k_block, i_block).
        # In the 4x4 Strassen, the result M_s is a 2x2 matrix.
        # It contributes to the C blocks.
        # Which C block?
        # The formula C = ... sums the M_s to form C.
        # But in the tensor decomposition, we are adding rank-1 terms directly.
        # The term (A_s B_s) contributes to the entire C.
        # No, A_s and B_s are specific linear combinations.
        # The product A_s B_s is a 2x2 matrix.
        # This 2x2 matrix is added to specific C blocks with coefficients.
        # BUT, in the rank-1 decomposition of the multiplication tensor,
        # each rank-1 term corresponds to a specific "path" i-j-k.
        # The Strassen algorithm replaces the 16 paths with 7 products.
        # Each product is itself a 2x2 multiplication (7 paths).
        # So the 49 terms cover all the contributions.
        # The W part of the term must map to the correct C indices.
        # The W_2x2 from the base Strassen term maps local (x,u) to global (k,i).
        # Local x is col of B block, local u is row of A block.
        # Global k = k_offset + bc*2 + x ?
        # Global i = br*2 + u ?
        # This assumes the result block is at (bc, br) in the C grid?
        # No.
        # Let's trace indices carefully.
        # A block at (br, bc) has rows br*2+u, cols bc*2+v.
        # B block at (br, bc) has rows br*2+v, cols bc*2+x.
        # Product element (u, x) is sum_v A[u,v] B[v,x].
        # This element contributes to C element (i, k) where i=u, k=x?
        # No.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # If we are in block (br, bc) for A and (br, bc) for B?
        # No, the blocks must align on j.
        # A block col bc -> j = bc*2+v.
        # B block row br -> j = br*2+v.
        # So we need bc = br?
        # In Strassen, we form A_s and B_s as 2x2 matrices.
        # A_s[u, v] is a linear combo of A blocks.
        # B_s[v, x] is a linear combo of B blocks.
        # The product C_s[u, x] = sum_v A_s[u, v] B_s[v, x].
        # This C_s is a 2x2 matrix.
        # Where does it go in the final C?
        # It is added to C with a coefficient.
        # But in the tensor decomposition, we don't add it later.
        # The rank-1 terms ARE the decomposition.
        # So the W part of the rank-1 term must point to the correct C indices.
        # The C indices are determined by the global i and k.
        # i = global row of A = br*2 + u?
        # k = global col of B = bc*2 + x?
        # But wait.
        # If A_s uses block (0,0) and B_s uses block (0,0),
        # then u is row in A(0,0), v is col in A(0,0)/row in B(0,0), x is col in B(0,0).
        # i = 0*2 + u = u.
        # k = 0*2 + x = x.
        # So C[u, x] gets the contribution.
        # If A_s uses block (1,1) and B_s uses block (1,1),
        # i = 1*2 + u = 2+u.
        # k = 1*2 + x = 2+x.
        # So C[2+u, 2+x] gets the contribution.
        
        # So, for a specific term in the expansion of A_s B_s:
        # We sum over the blocks used in A_s and B_s.
        # For each pair of blocks (a_block, b_block) with coeffs ca, cb:
        #   For each local u, v, x:
        #     val = ca * cb * U_2x2[u][v] * V_2x2[v][x] * W_2x2[x][u]
        #     global_i = a_block.row * 2 + u
        #     global_j = a_block.col * 2 + v  (must match b_block.row * 2 + v)
        #     global_k = b_block.col * 2 + x
        #     U[global_i*m+global_j] += val_part_U
        #     V[global_j*p+global_k] += val_part_V
        #     W[global_k*n+global_i] += val_part_W
        
        # This requires iterating over all combinations of blocks in A_s and B_s.
        
        # Reset col increment logic:
        # We are adding ONE rank-1 term to the global U, V, W.
        # So we iterate over all (a_block, b_block, u, v, x) that contribute to this term.
        
        for a_br, a_bc in a_coeffs:
            ca = a_coeffs[(a_br, a_bc)]
            for b_br, b_bc in b_coeffs:
                cb = b_coeffs[(b_br, b_bc)]
                # Check if j matches
                # j from A is a_bc*2+v
                # j from B is b_br*2+v
                # So we need a_bc == b_br
                if a_bc != b_br:
                    continue
                
                for u in range(2):
                    for v in range(2):
                        for x in range(2):
                            u_val = U_2x2[u][v]
                            v_val = V_2x2[v][x]
                            w_val = W_2x2[x][u]
                            
                            if u_val == 0 or v_val == 0 or w_val == 0:
                                continue
                            
                            coeff = ca * cb
                            val_u = coeff * u_val
                            val_v = coeff * v_val
                            val_w = coeff * w_val
                            
                            global_i = a_br * 2 + u
                            global_j = a_bc * 2 + v
                            global_k = k_offset + b_bc * 2 + x
                            
                            U[global_i * m + global_j][col] += val_u
                            V[global_j * p + global_k][col] += val_v
                            W[global_k * n + global_i][col] += val_w
        
        col += 1
        
    for a_coeffs, b_coeffs in strassen_4x4:
        for term in strassen_2x2_terms:
            add_strassen_4x4_term(a_coeffs, b_coeffs, term, k_offset=0)
            
    # --- Part 2: 4x4 x 4x3 (columns 4..6 of B) ---
    # Schoolbook for 2x2 x 2x3 blocks
    
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: Top (rows 0-1, cols 4-6), Bot (rows 2-3, cols 4-6)
    # We iterate over A blocks and B blocks (Top/Bot)
    
    # A block (ar, ac) is 2x2.
    # B block (br, bc) where br is 0 or 1 (Top/Bot), bc is 0..2 (cols 4-6)
    # Actually, let's just iterate over global indices for Part 2.
    # i in 0..3, j in 0..3, k in 4..6
    
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
