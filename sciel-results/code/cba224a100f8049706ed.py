def construct():
    n, m, p = 4, 4, 7
    rank = 104
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a scalar product term
    # u_idx, v_idx, w_indices (list of (idx, coeff))
    def add_term(u_idx, v_idx, w_coeffs, u_coeff=1, v_coeff=1):
        nonlocal col
        U[u_idx][col] += u_coeff
        V[v_idx][col] += v_coeff
        for w_idx, w_coeff in w_coeffs:
            W[w_idx][col] += w_coeff
        col += 1

    # Define A block indices
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A12: (0,2), (0,3), (1,2), (1,3)
    # A21: (2,0), (2,1), (3,0), (3,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    
    # Define B block indices (cols 0-3)
    # B11: (0,0), (0,1), (1,0), (1,1)
    # B12: (0,2), (0,3), (1,2), (1,3)
    # B21: (2,0), (2,1), (3,0), (3,1)
    # B22: (2,2), (2,3), (3,2), (3,3)
    
    # We define the 7 products as lists of (L_block, R_block, signs)
    # L_block is a string like "A11", R_block is "B11+B22"
    # But it's easier to define the linear combinations explicitly.
    
    # M1 = A11 * (B11 + B22)
    # L = A11, R = B11 + B22
    # Reconstruction: C11 (+1), C22 (+1)
    
    # We'll write a generic function to process a product
    # process_product(L_def, R_def, recon_coeffs)
    # L_def: list of (block, sign) for the 2x2 matrix
    # R_def: list of (block, sign) for the 2x2 matrix
    # recon_coeffs: dict {block: coeff}
    
    def process_product(L_blocks, R_blocks, recon):
        nonlocal col
        # L_blocks is a 2x2 list of lists of (block_name, sign)
        # R_blocks is a 2x2 list of lists of (block_name, sign)
        
        # For each element (a,d) in the 2x2 result
        for a in range(2):
            for d in range(2):
                # Sum over b
                for b in range(2):
                    # Get L[a,b] and R[b,d]
                    # L[a,b] is a sum of A terms
                    # R[b,d] is a sum of B terms
                    for (la_block, la_sign) in L_blocks[a][b]:
                        for (rb_block, rb_sign) in R_blocks[b][d]:
                            # Get the actual indices
                            # la_block is one of A11, A12, A21, A22
                            # rb_block is one of B11, B12, B21, B22
                            
                            # Map block to base indices
                            # A11: row 0, col 0. Offset (0,0)
                            # A12: row 0, col 2. Offset (0,2)
                            # A21: row 2, col 0. Offset (2,0)
                            # A22: row 2, col 2. Offset (2,2)
                            
                            # We need the specific element within the 2x2 block.
                            # The block definition L_blocks[a][b] contains terms for the (a,b) position of the 2x2 block.
                            # So if la_block is "A11", the element is A[row_base + a, col_base + b]?
                            # No. L_blocks is defined for the 2x2 structure.
                            # If L_blocks[0][0] contains ("A11", 1), it means the (0,0) element of the 2x2 matrix L is A11[0,0]?
                            # No, A11 is a 2x2 block.
                            # In Strassen, L is a 2x2 matrix of blocks? No, L is a 2x2 matrix of scalars?
                            # No, in the block Strassen, L is a 2x2 matrix where each entry is a 2x2 matrix (block).
                            # So L[0,0] is A11.
                            # So the element (a,b) of the 2x2 *block* matrix is the block A11.
                            # But we are decomposing into scalar products.
                            # So we need to iterate through the elements of the block A11.
                            
                            # Let's refine.
                            # L_blocks[a][b] gives the block name.
                            # The block is 2x2.
                            # We need to iterate through the 4 elements of the block?
                            # No. The matrix multiplication is of 2x2 block matrices.
                            # $M = L R$.
                            # $M[a,d] = \sum_b L[a,b] R[b,d]$.
                            # $L[a,b]$ is a 2x2 matrix. $R[b,d]$ is a 2x2 matrix.
                            # Their product is a 2x2 matrix.
                            # So $M[a,d]$ is a 2x2 matrix.
                            # This matches our previous logic: $M_k$ is a 2x2 matrix.
                            
                            # So, for a fixed $a, b, d$:
                            # We multiply block $L_{ab}$ by block $R_{bd}$.
                            # This results in 4 scalar products (2x2 mat mult).
                            # We need to add these 4 scalar products.
                            
                            # Let's get the indices for the blocks.
                            def get_block_indices(block_name):
                                if block_name == "A11": return (0,0)
                                if block_name == "A12": return (0,2)
                                if block_name == "A21": return (2,0)
                                if block_name == "A22": return (2,2)
                                if block_name == "B11": return (0,0)
                                if block_name == "B12": return (0,2)
                                if block_name == "B21": return (2,0)
                                if block_name == "B22": return (2,2)
                                return None
                            
                            # We need to know the element position within the block.
                            # The block multiplication $L_{ab} R_{bd}$ produces a 2x2 matrix.
                            # Let the elements of $L_{ab}$ be $l_{i,j}$ and $R_{bd}$ be $r_{k,l}$.
                            # Result element $(i,l)$ is $\sum_j l_{i,j} r_{j,l}$.
                            
                            # We need to map the block name to the actual A/B indices.
                            # A block "A11" at position (row_base, col_base) has elements A[row_base+i, col_base+j].
                            # B block "B11" at position (row_base, col_base) has elements B[row_base+k, col_base+l].
                            
                            # The result of the block product contributes to the (a,d) block of M.
                            # This block (a,d) of M corresponds to a specific 2x2 region in C.
                            # We need to map (a,d) to C offsets.
                            # M is 2x2 blocks.
                            # Block (0,0) -> C[0:2, 0:2]
                            # Block (0,1) -> C[0:2, 2:4]
                            # Block (1,0) -> C[2:4, 0:2]
                            # Block (1,1) -> C[2:4, 2:4]
                            
                            # So if we are computing element (i,l) of the block product,
                            # it contributes to C[row_base_c + i, col_base_c + l].
                            
                            # We also have reconstruction coefficients.
                            # recon is a dict {block_name: coeff}.
                            # block_name is "11", "12", "21", "22".
                            # The block (a,d) corresponds to:
                            # (0,0) -> "11"
                            # (0,1) -> "12"
                            # (1,0) -> "21"
                            # (1,1) -> "22"
                            
                            # So for each scalar product in the block multiplication,
                            # we add it to the corresponding C elements based on recon.
                            
                            # Let's implement the inner loop.
                            l_row, l_col = get_block_indices(la_block)
                            r_row, r_col = get_block_indices(rb_block)
                            
                            # Iterate over the 2x2 block multiplication
                            for i in range(2):
                                for l in range(2):
                                    # Sum over j
                                    w_coeffs = []
                                    # Determine C block for (a,d)
                                    c_block_name = f"{a+1}{d+1}"
                                    coeff = recon.get(c_block_name, 0)
                                    if coeff == 0: continue
                                    
                                    # C offset for this block
                                    c_row_base = 0 if a == 0 else 2
                                    c_col_base = 0 if d == 0 else 2
                                    
                                    for j in range(2):
                                        # A index
                                        a_idx = (l_row + i) * 4 + (l_col + j)
                                        # B index
                                        b_idx = (r_row + j) * 7 + (r_col + l)
                                        
                                        # C indices
                                        c_idx = (c_row_base + i) + (c_col_base + l) * 4 # Wait, W is (p*n, rank).
                                        # W index: k*n + i. k is col, i is row.
                                        # c_col = c_col_base + l
                                        # c_row = c_row_base + i
                                        # w_idx = c_col * 4 + c_row
                                        
                                        w_idx = (c_col_base + l) * 4 + (c_row_base + i)
                                        
                                        # Coefficients
                                        u_c = la_sign
                                        v_c = rb_sign
                                        w_c = coeff * u_c * v_c
                                        
                                        # We add this term.
                                        # But wait, we are iterating i, l, j.
                                        # This generates 4 scalar products for each (a,b,d) block pair.
                                        # We need to add them to U, V, W.
                                        
                                        # Since we are inside a loop that generates unique scalar products,
                                        # we should call add_term for each one.
                                        
                                        # However, add_term increments col.
                                        # We need to ensure we don't exceed rank.
                                        
                                        add_term(a_idx, b_idx, [(w_idx, w_c)], u_c, v_c)
    
    # Define the 7 products
    # L_blocks and R_blocks are 2x2 lists of lists of (block, sign)
    
    # M1 = A11 * (B11 + B22)
    L1 = [[ [("A11", 1)] ], [ [("A11", 1)] ]] # This is wrong structure.
    
    # Let's use a simpler representation.
    # L is a 2x2 matrix of block names.
    # If a block is a sum, we handle it by listing multiple terms.
    
    # M1:
    # L[0,0]=A11, L[0,1]=0, L[1,0]=0, L[1,1]=0? No.
    # L is A11. So L[0,0]=A11[0,0]? No.
    # In block Strassen, L is a 2x2 matrix of blocks.
    # For M1, L is just A11 in the (0,0) position? No.
    # M1 = A11 * (B11 + B22).
    # This is not a standard 2x2 block multiplication.
    # It's a product of a 2x2 block A11 and a 2x2 block (B11+B22).
    # So L is a 1x1 matrix of blocks? No.
    # The Strassen algorithm operates on 2x2 block matrices.
    # A = [[A11, A12], [A21, A22]].
    # B = [[B11, B12], [B21, B22]].
    # M1 = A11 * (B11 + B22).
    # Here, A11 is a block. (B11+B22) is a block.
    # So M1 is a single block product.
    # It contributes to the (0,0) and (1,1) positions of the result?
    # No, M1 is just a block.
    # The reconstruction adds M1 to C11 and C22.
    # So M1 is a 2x2 matrix (block).
    # So we just need to compute the product of block A11 and block (B11+B22).
    
    # So, for each M_k, we identify the block L and block R.
    # Then we compute the 2x2 block multiplication.
    
    # M1: L=A11, R=B11+B22. Recon: C11(+1), C22(+1).
    # M2: L=A11+A22, R=B22. Recon: C21(+1), C22(-1).
    # M3: L=A12-A22, R=B21+B22. Recon: C12(+1), C22(+1).
    # M4: L=A11, R=B11-B21. Recon: C11(+1), C21(+1).
    # M5: L=A11+A12, R=B21. Recon: C11(-1), C12(+1).
    # M6: L=A21-A11, R=B11+B12. Recon: C22(+1).
    # M7: L=A22-A12, R=B12+B21. Recon: C11(+1).
    
    # We define these as lists of (L_block, L_sign) and (R_block, R_sign).
    
    products = [
        # (L_terms, R_terms, recon)
        ( [("A11", 1)], [("B11", 1), ("B22", 1)], {"11": 1, "22": 1} ),
        ( [("A11", 1), ("A22", 1)], [("B22", 1)], {"21": 1, "22": -1} ),
        ( [("A12", 1), ("A22", -1)], [("B21", 1), ("B22", 1)], {"12": 1, "22": 1} ),
        ( [("A11", 1)], [("B11", 1), ("B21", -1)], {"11": 1, "21": 1} ),
        ( [("A11", 1), ("A12", 1)], [("B21", 1)], {"11": -1, "12": 1} ),
        ( [("A21", 1), ("A11", -1)], [("B11", 1), ("B12", 1)], {"22": 1} ),
        ( [("A22", 1), ("A12", -1)], [("B12", 1), ("B21", 1)], {"11": 1} ),
    ]
    
    def get_block_base(block_name):
        if block_name == "A11": return (0,0)
        if block_name == "A12": return (0,2)
        if block_name == "A21": return (2,0)
        if block_name == "A22": return (2,2)
        if block_name == "B11": return (0,0)
        if block_name == "B12": return (0,2)
        if block_name == "B21": return (2,0)
        if block_name == "B22": return (2,2)
        return (0,0)
        
    for L_terms, R_terms, recon in products:
        # L_terms is a list of (block, sign).
        # This represents the 2x2 block L.
        # Wait, L is a 2x2 matrix.
        # In Strassen, L is formed by adding/subtracting blocks.
        # e.g. L = A11 + A22.
        # This means L[0,0] = A11[0,0] + A22[0,0]?
        # No. A11 is a 2x2 block. A22 is a 2x2 block.
        # A11 + A22 is a 2x2 block.
        # So L is a single 2x2 block.
        # So we just multiply this 2x2 block L by the 2x2 block R.
        
        # So we iterate over the elements of the 2x2 block multiplication.
        
        # L is sum of blocks. R is sum of blocks.
        # L R = (sum l_i L_i) (sum r_j R_j) = sum sum l_i r_j L_i R_j.
        
        for (l_block, l_sign) in L_terms:
            for (r_block, r_sign) in R_terms:
                l_row, l_col = get_block_base(l_block)
                r_row, r_col = get_block_base(r_block)
                
                # Compute 2x2 block product
                for i in range(2):
                    for k in range(2):
                        # Result element (i,k) of the block product
                        # This contributes to C based on recon.
                        # Which block of C does this correspond to?
                        # The product M_k is a single 2x2 block.
                        # It is added to specific C blocks.
                        # The element (i,k) of M_k adds to the element (i,k) of each C block in recon.
                        
                        for c_block, c_sign in recon.items():
                            # Map c_block to C offset
                            if c_block == "11":
                                c_row_base, c_col_base = 0, 0
                            elif c_block == "12":
                                c_row_base, c_col_base = 0, 2
                            elif c_block == "21":
                                c_row_base, c_col_base = 2, 0
                            else: # 22
                                c_row_base, c_col_base = 2, 2
                                
                            # Sum over j for the dot product
                            for j in range(2):
                                a_idx = (l_row + i) * 4 + (l_col + j)
                                b_idx = (r_row + j) * 7 + (r_col + k)
                                c_row = c_row_base + i
                                c_col = c_col_base + k
                                w_idx = c_col * 4 + c_row
                                
                                u_c = l_sign
                                v_c = r_sign
                                w_c = c_sign * u_c * v_c
                                
                                add_term(a_idx, b_idx, [(w_idx, w_c)], u_c, v_c)
    
    # Now handle the remaining 3 columns (4-6) with schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                u_idx = i * 4 + j
                v_idx = j * 7 + k
                w_idx = k * 4 + i
                add_term(u_idx, v_idx, [(w_idx, 1)], 1, 1)
                
    return {"U": U, "V": V, "W": W}
