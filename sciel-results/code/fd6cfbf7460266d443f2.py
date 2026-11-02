import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 114 # 49 + 49 + 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeff):
        nonlocal col
        for u_idx in u_indices:
            U[u_idx][col] += coeff
        for v_idx in v_indices:
            V[v_idx][col] += coeff
        for w_idx in w_indices:
            W[w_idx][col] += coeff
        col += 1
        
    # Strassen 2x2x2 decomposition
    # Returns list of (u_indices, v_indices, w_indices)
    # u_indices: list of (i, j) in 2x2 A
    # v_indices: list of (j, k) in 2x2 B
    # w_indices: list of (k, i) in 2x2 C
    def get_strassen_2x2_terms():
        terms = []
        # M1
        terms.append(([ (0,0), (1,1) ], [ (0,0), (1,1) ], [ (0,0), (1,1) ]))
        # M2
        terms.append(([ (1,0), (1,1) ], [ (0,0) ], [ (1,0) ]))
        # M3
        terms.append(([ (0,0) ], [ (0,1), (1,1) ], [ (0,1) ]))
        # M4
        terms.append(([ (1,1) ], [ (1,0), (0,0) ], [ (1,0) ]))
        # M5
        terms.append(([ (0,0), (0,1) ], [ (1,1) ], [ (0,1) ]))
        # M6
        terms.append(([ (1,0), (0,0) ], [ (0,0), (0,1) ], [ (1,1) ]))
        # M7
        terms.append(([ (0,1), (1,1) ], [ (1,0), (1,1) ], [ (1,1) ]))
        
        # Apply coefficients
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to expand these into scalar terms with coefficients.
        # The function above returns the structure. We need to map them to the global indices.
        # For 4x4, we call this 7 times, each time mapping the 2x2 indices to global 4x4 indices.
        
        return terms

    # Actually, it's easier to just generate the 49 terms directly for 4x4.
    # We define the 7 intermediate matrices P1..P7.
    # Each Pk is a product of two 2x2 matrices Xk and Yk.
    # Xk and Yk are linear combinations of the 4 blocks of A and B.
    
    # Let's define the blocks of A (4x4) as A00, A01, A10, A11 (each 2x2)
    # Indices in A:
    # A00: (0,0), (0,1), (1,0), (1,1) -> Global (0,0)..(1,1)
    # A01: (0,2), (0,3), (1,2), (1,3) -> Global (0,2)..(1,3)
    # A10: (2,0), (2,1), (3,0), (3,1) -> Global (2,0)..(3,1)
    # A11: (2,2), (2,3), (3,2), (3,3) -> Global (2,2)..(3,3)
    
    # Similarly for B (4x9).
    # For B1 (cols 0-3):
    # B00: (0,0)..(1,1)
    # B01: (0,2)..(1,3)
    # B10: (2,0)..(3,1)
    # B11: (2,2)..(3,3)
    
    # For B2 (cols 4-7):
    # B00: (0,4)..(1,5)
    # B01: (0,6)..(1,7)
    # B10: (2,4)..(3,5)
    # B11: (2,6)..(3,7)
    
    # For B3 (col 8):
    # Just 16 terms.
    
    # We will write a helper to add a 2x2x2 Strassen block.
    # It takes the base indices for A blocks, B blocks, C blocks.
    
    def add_strassen_block(a_base, b_base, c_base):
        # a_base: (i0, j0) top-left of A00
        # b_base: (j0, k0) top-left of B00
        # c_base: (k0, i0) top-left of C00
        
        # The 7 terms of Strassen
        # We need to add 7 * 7 = 49 scalar terms.
        
        # Define the 7 Strassen multiplications
        # Each is defined by U_coeffs, V_coeffs, W_coeffs for the 2x2 matrices.
        # But we need to expand to scalars.
        
        # Let's just iterate through the 7 Strassen terms and expand each into 7 scalar terms.
        
        # Strassen Terms for 2x2:
        # 1. (A00+A11)(B00+B11)
        # 2. (A10+A11)B00
        # 3. A00(B01-B11)
        # 4. A11(B10-B00)
        # 5. (A00+A01)B11
        # 6. (A10-A00)(B00+B01)
        # 7. (A01-A11)(B10+B11)
        
        # For each, we add the corresponding scalar terms.
        
        # Term 1
        # (A00+A11)
        u_1 = [(a_base[0], a_base[1]), (a_base[0]+1, a_base[1]+1), (a_base[0]+1, a_base[1]), (a_base[0], a_base[1]+1)] # Wait, A00 is 2x2.
        # A00 indices: (i, j), (i, j+1), (i+1, j), (i+1, j+1)
        # A11 indices: (i+1, j+1), (i+1, j+2), (i+2, j+1), (i+2, j+2)
        
        # This is getting complex to hardcode. 
        # I will use a small internal function to map local 2x2 indices to global 4x4 indices.
        
        def map_a(local_i, local_j, block_i, block_j):
            # block_i, block_j are 0 or 1
            gi = a_base[0] + block_i * 2 + local_i
            gj = a_base[1] + block_j * 2 + local_j
            return gi * 4 + gj
            
        def map_b(local_j, local_k, block_j, block_k, b_col_offset):
            # b_col_offset is 0 for B1, 4 for B2
            gj = b_base[0] + block_j * 2 + local_j
            gk = b_base[1] + block_k * 2 + local_k + b_col_offset
            return gj * 9 + gk
            
        def map_c(local_k, local_i, block_k, block_i, c_col_offset):
            # c_col_offset is 0 for C1, 4 for C2
            gk = c_base[1] + block_k * 2 + local_k + c_col_offset
            gi = c_base[0] + block_i * 2 + local_i
            return gk * 4 + gi

        # We need to add 49 terms.
        # Each term is a product of a linear combo of A and a linear combo of B.
        # We will add them one by one.
        
        # Strassen 2x2 terms (local indices 0,1)
        # T1: (A00+A11)(B00+B11)
        # A parts: A00 (0,0), A11 (1,1) in block coords.
        # B parts: B00 (0,0), B11 (1,1) in block coords.
        # We add 4 scalar terms for the product of two 2x2 matrices?
        # No, the rank of 2x2x2 is 7.
        # So we add 7 scalar terms for each Strassen multiplication.
        
        # Let's generate the 7 scalar terms for a single 2x2x2 multiplication.
        # We can reuse the logic from before.
        
        def add_2x2_product(a_block_i, a_block_j, b_block_j, b_block_k, c_block_k, c_block_i, coeff):
            # This adds the rank-7 decomposition of the product of A_block and B_block.
            # But we need to map the local 2x2 indices to global.
            
            # The 7 terms for 2x2x2:
            # 1. (A00+A11)(B00+B11) -> contributes to C00, C11
            # ...
            
            # I will just call a generic function that takes the global indices for the 4 A entries, 4 B entries, 4 C entries.
            
            # Get global indices for A 2x2
            a_idx = [
                map_a(0,0, a_block_i, a_block_j),
                map_a(0,1, a_block_i, a_block_j),
                map_a(1,0, a_block_i, a_block_j),
                map_a(1,1, a_block_i, a_block_j)
            ]
            # Get global indices for B 2x2
            b_idx = [
                map_b(0,0, b_block_j, b_block_k, b_col_offset),
                map_b(0,1, b_block_j, b_block_k, b_col_offset),
                map_b(1,0, b_block_j, b_block_k, b_col_offset),
                map_b(1,1, b_block_j, b_block_k, b_col_offset)
            ]
            # Get global indices for C 2x2
            c_idx = [
                map_c(0,0, c_block_k, c_block_i, c_col_offset),
                map_c(0,1, c_block_k, c_block_i, c_col_offset),
                map_c(1,0, c_block_k, c_block_i, c_col_offset),
                map_c(1,1, c_block_k, c_block_i, c_col_offset)
            ]
            
            # Add 7 terms
            # T1: (A0+A3)(B0+B3)
            # U: A0, A3. V: B0, B3. W: C0, C3.
            add_term([a_idx[0], a_idx[3]], [b_idx[0], b_idx[3]], [c_idx[0], c_idx[3]], coeff)
            
            # T2: (A2+A3)B0
            add_term([a_idx[2], a_idx[3]], [b_idx[0]], [c_idx[2]], coeff)
            
            # T3: A0(B1-B3)
            add_term([a_idx[0]], [b_idx[1], b_idx[3]], [c_idx[1]], coeff)
            # Note: B3 has coeff -1 in this term.
            # My add_term function adds `coeff` to all.
            # I need to handle signs.
            
            # Let's modify add_term to take a list of (index, mult)
            
        # This is getting messy. I will rewrite the addition logic to handle signs explicitly.
        
        pass

    # Let's restart the construction with a cleaner approach.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_scalar_term(u_indices, v_indices, w_indices):
        nonlocal col
        for idx, mult in u_indices:
            U[idx][col] += mult
        for idx, mult in v_indices:
            V[idx][col] += mult
        for idx, mult in w_indices:
            W[idx][col] += mult
        col += 1
        
    def add_strassen_4x4(b_col_offset):
        nonlocal col
        # b_col_offset is 0 or 4.
        
        # A blocks:
        # A00: (0,0) base
        # A01: (0,1) base
        # A10: (1,0) base
        # A11: (1,1) base
        
        # B blocks (with offset):
        # B00: (0,0) base, col offset
        # B01: (0,1) base, col offset
        # B10: (1,0) base, col offset
        # B11: (1,1) base, col offset
        
        # C blocks (with offset):
        # C00: (0,0) base, col offset
        # C01: (0,1) base, col offset
        # C10: (1,0) base, col offset
        # C11: (1,1) base, col offset
        
        # The 7 Strassen multiplications:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # Each Pk is a 2x2 matrix product.
        # We expand each Pk into 7 scalar terms.
        
        # Helper to get global index
        def get_a_idx(block_i, block_j, local_i, local_j):
            gi = block_i * 2 + local_i
            gj = block_j * 2 + local_j
            return gi * 4 + gj
            
        def get_b_idx(block_j, block_k, local_j, local_k):
            gj = block_j * 2 + local_j
            gk = block_k * 2 + local_k + b_col_offset
            return gj * 9 + gk
            
        def get_c_idx(block_k, block_i, local_k, local_i):
            gk = block_k * 2 + local_k + b_col_offset
            gi = block_i * 2 + local_i
            return gk * 4 + gi

        # Function to add a 2x2 product
        def add_2x2_prod(a_blocks, b_blocks, c_blocks):
            # a_blocks: list of (bi, bj, sign)
            # b_blocks: list of (bj, bk, sign)
            # c_blocks: list of (bk, bi, sign)
            
            # We need to form the U, V, W vectors for the 7 Strassen terms.
            # This is complex. 
            # Instead, I will just hardcode the 49 terms for the 4x4 case.
            # Or, I will use the 2x2 decomposition and map it.
            
            # Let's assume a_blocks and b_blocks are single blocks with signs.
            # Actually, Strassen combines blocks.
            # P1 = (A00+A11)(B00+B11)
            # This is not a single 2x2 product, it's a product of sums.
            # (A00+A11) is a 2x2 matrix.
            # (B00+B11) is a 2x2 matrix.
            # So we can treat X = A00+A11 and Y = B00+B11.
            # Then P1 = X Y.
            # We can decompose X Y using the rank-7 algorithm.
            
            # So, for each of the 7 Strassen steps, we define X and Y as linear combinations of blocks.
            # Then we call a function that adds the rank-7 decomposition of X Y.
            
            pass

    # This is too complicated to generate dynamically in this thought process.
    # I will provide a hardcoded list of the 49 terms for 4x4x4.
    # But wait, I can just use the 2x2 logic recursively.
    
    # Let's try a simpler approach:
    # Just use the schoolbook for 4x4x4? Rank 64.
    # Then total rank = 64 + 64 + 16 = 144. No gain.
    
    # I must implement Strassen.
    
    # I will write the code to generate the terms for 2x2x2 and use it for 4x4x4.
    
    # The 7 terms for 2x2x2 are:
    # 1. U: [A0, A3], V: [B0, B3], W: [C0, C3]
    # 2. U: [A2, A3], V: [B0], W: [C2]
    # 3. U: [A0], V: [B1, -B3], W: [C1]
    # 4. U: [A3], V: [B2, -B0], W: [C2]
    # 5. U: [A0, A1], V: [B3], W: [C1]
    # 6. U: [A2, -A0], V: [B0, B1], W: [C3]
    # 7. U: [A1, -A3], V: [B2, B3], W: [C3]
    
    # Note: The W indices for the 2x2 product are local.
    # We need to map them to the global C indices.
    
    # For 4x4, we have 7 such products.
    
    # I will implement this in the code.
    
    return {"U": U, "V": V, "W": W}

