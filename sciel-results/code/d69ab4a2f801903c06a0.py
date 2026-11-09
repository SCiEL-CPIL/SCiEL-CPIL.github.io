import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (32, rank)
    
    # Rank is 98
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # term_u: dict mapping (i, j) to coeff
    # term_v: dict mapping (j, k) to coeff
    # term_w: dict mapping (k, i) to coeff
    # col: column index in U, V, W
    def add_term(term_u, term_v, term_w, col):
        for (i, j), c in term_u.items():
            U[i * m + j][col] = c
        for (j, k), c in term_v.items():
            V[j * p + k][col] = c
        for (k, i), c in term_w.items():
            W[k * n + i][col] = c

    # Strassen 2x2x2 decomposition
    # Returns a list of 7 tuples (term_u, term_v, term_w)
    # where term_u, term_v, term_w are dicts for the 2x2, 2x2, 2x2 blocks
    # The indices in the dicts are local 0,1
    def strassen_2x2_terms():
        terms = []
        
        # P1 = (A00 + A11)(B00 + B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {} # Will be determined by combination
        # Actually, the rank-1 term is defined by the product of the linear forms.
        # The W part is not a single rank-1 term in the same way.
        # The tensor is sum of rank-1 terms.
        # A rank-1 term is u * v * w where u is a vector in A-space, v in B-space, w in C-space.
        # For Strassen, the 7 products are intermediate.
        # The final C is a linear combination of these products.
        # So we need to expand the final C into rank-1 terms.
        
        # Let's just generate the 7 intermediate products as rank-1 terms in the "block" space.
        # Then substitute the block entries with their linear combinations.
        
        # This is getting complicated to track manually.
        # Let's use a simpler approach: 
        # The rank of 4x4x4 is 49.
        # I will hardcode the 49 rank-1 terms if possible, or generate them via a known library/logic.
        # Since I can't use external libraries for the decomposition coefficients easily, I will derive them.
        
        # Alternative: Use the rank 112 Strassen decomposition which is easier to derive.
        # Rank 112 is still an improvement over 128.
        # Let's stick to Rank 112 to ensure correctness.
        
        # Rank 112 Strategy:
        # Split 4x4 into 2x2 blocks.
        # 7 Strassen products of 2x2 blocks.
        # Each 2x2 product is done via schoolbook (rank 4) or Strassen (rank 7)?
        # If schoolbook, rank = 7 * 4 = 28 for 4x4x4. Total 56 for 4x4x8.
        # Wait, 2x2x2 schoolbook rank is 4.
        # 7 * 4 = 28.
        # Is rank 28 possible for 4x4x4? No, lower bound is 25 (Winograd).
        # Strassen 2x2x2 is rank 7.
        # So 7 * 7 = 49 is the Strassen rank for 4x4x4.
        
        # Let's implement the 49 rank decomposition properly.
        # I will define the 7 Strassen products for 2x2x2.
        # Then I will treat each product as a scalar multiplication in the block algebra.
        # Then I will expand each block multiplication into 7 rank-1 terms.
        
        pass

    # Let's restart with a clear plan for Rank 112 (7 Strassen products of schoolbook 2x2x2).
    # This is safer and still better than 128.
    
    # 2x2x2 Schoolbook terms:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # For 4x4x4 Strassen:
    # 7 products M1..M7 of 2x2 blocks.
    # Each Mi is computed via schoolbook (4 terms).
    # Total 28 terms for 4x4x4.
    # Total 56 terms for 4x4x8.
    # Rank 56 is much better than 128.
    
    # Let's implement Rank 56.
    
    # Clear U, V, W
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add a 2x2x2 schoolbook term
    # a_idx: (i, j) in 2x2 block A
    # b_idx: (j, k) in 2x2 block B
    # c_idx: (i, k) in 2x2 block C
    # block_a_offset: (i0, j0)
    # block_b_offset: (j0, k0)
    # block_c_offset: (i0, k0)
    # coeff: coefficient
    
    def add_2x2_term(a_idx, b_idx, c_idx, a_off, b_off, c_off, coeff, col):
        i_a, j_a = a_idx
        j_b, k_b = b_idx
        i_c, k_c = c_idx
        
        i = a_off[0] + i_a
        j = a_off[1] + j_a
        U[i * m + j][col] = coeff
        
        j = b_off[0] + j_b
        k = b_off[1] + k_b
        V[j * p + k][col] = coeff
        
        i = c_off[0] + i_c
        k = c_off[1] + k_c
        W[k * n + i][col] = coeff

    # Strassen 2x2x2 products (linear combinations of blocks)
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Block indices for 4x4:
    # A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
    # B11: (0,0), B12: (0,1), B21: (1,0), B22: (1,1)
    # C11: (0,0), C12: (0,1), C21: (1,0), C22: (1,1)
    
    # For each Strassen product, we have a linear combination of A blocks and B blocks.
    # We need to expand (Sum a_ij A_block_ij) * (Sum b_jk B_block_jk) into schoolbook terms.
    # This results in multiple rank-1 terms.
    
    # To keep it simple, I will just use the Rank 112 decomposition (Strassen 2x2x2 for the 2x2 blocks).
    # Wait, Rank 56 is Strassen 4x4x4 using Schoolbook 2x2x2.
    # Rank 112 is Strassen 4x4x4 using Strassen 2x2x2.
    # Rank 56 is better. Let's do Rank 56.
    
    # For each of the 7 Strassen products, we compute the product of two linear combinations of 2x2 blocks.
    # Let U_A be the linear combination of A blocks for this product.
    # Let V_B be the linear combination of B blocks for this product.
    # The product U_A * V_B is a 2x2 matrix product.
    # We decompose this 2x2 matrix product into 4 schoolbook rank-1 terms.
    # So each Strassen product contributes 4 rank-1 terms.
    # Total 7 * 4 = 28 terms for 4x4x4.
    # Total 56 terms for 4x4x8.
    
    # Define the 7 Strassen products as lists of (block_idx, coeff) for A and B.
    # A blocks: 11, 12, 21, 22
    # B blocks: 11, 12, 21, 22
    
    strassen_products = [
        # M1
        ([ ( (0,0), 1 ), ( (1,1), 1 ) ], [ ( (0,0), 1 ), ( (1,1), 1 ) ]),
        # M2
        ([ ( (1,0), 1 ), ( (1,1), 1 ) ], [ ( (0,0), 1 ) ]),
        # M3
        ([ ( (0,0), 1 ) ], [ ( (0,1), 1 ), ( (1,1), -1 ) ]),
        # M4
        ([ ( (1,1), 1 ) ], [ ( (1,0), 1 ), ( (0,0), -1 ) ]),
        # M5
        ([ ( (0,0), 1 ), ( (0,1), 1 ) ], [ ( (1,1), 1 ) ]),
        # M6
        ([ ( (1,0), 1 ), ( (0,0), -1 ) ], [ ( (0,0), 1 ), ( (0,1), 1 ) ]),
        # M7
        ([ ( (0,1), 1 ), ( (1,1), -1 ) ], [ ( (1,0), 1 ), ( (1,1), 1 ) ])
    ]
    
    # For each product, expand into 4 schoolbook terms.
    # The product is (Sum a_u A_u) * (Sum b_v B_v)
    # = Sum_u Sum_v a_u b_v A_u B_v
    # Each A_u B_v is a 2x2x2 schoolbook product (4 terms).
    # So we add 4 terms for each pair (u, v).
    
    for a_comb, b_comb in strassen_products:
        for (a_block, a_coeff) in a_comb:
            for (b_block, b_coeff) in b_comb:
                coeff = a_coeff * b_coeff
                if coeff == 0: continue
                
                # Add 4 schoolbook terms for A_block * B_block
                # A_block is at offset a_block, B_block at offset b_block
                # The result contributes to C.
                # We need to map the local 2x2 indices to global.
                
                # Schoolbook terms for 2x2:
                # (0,0)*(0,0) -> C(0,0)
                # (0,1)*(1,0) -> C(0,0)
                # (0,0)*(0,1) -> C(0,1)
                # (0,1)*(1,1) -> C(0,1)
                # (1,0)*(0,0) -> C(1,0)
                # (1,1)*(1,0) -> C(1,0)
                # (1,0)*(0,1) -> C(1,1)
                # (1,1)*(1,1) -> C(1,1)
                
                # Local indices
                sb_terms = [
                    ((0,0), (0,0), (0,0)),
                    ((0,1), (1,0), (0,0)),
                    ((0,0), (0,1), (0,1)),
                    ((0,1), (1,1), (0,1)),
                    ((1,0), (0,0), (1,0)),
                    ((1,1), (1,0), (1,0)),
                    ((1,0), (0,1), (1,1)),
                    ((1,1), (1,1), (1,1))
                ]
                
                for (ia, ja), (jb, kb), (ic, kc) in sb_terms:
                    # Global indices
                    i = a_block[0] * 2 + ia
                    j = a_block[1] * 2 + ja
                    U[i * m + j][col] = coeff
                    
                    j = b_block[0] * 2 + jb
                    k = b_block[1] * 2 + kb
                    V[j * p + k][col] = coeff
                    
                    i = a_block[0] * 2 + ic
                    k = b_block[1] * 2 + kc
                    W[k * n + i][col] = coeff
                    
                    col += 1
    
    # This gives rank 56 for 4x4x4.
    # We need 4x4x8.
    # We repeat this for the second half of B.
    # The second half of B is columns 4-7.
    # In the above, B indices k were 0-3.
    # For the second half, we add 4 to k.
    
    # We need to reset col or continue?
    # The rank is 56 + 56 = 112?
    # No, the rank is the sum of the ranks of the independent multiplications.
    # So we add another 56 terms.
    
    # Let's re-run the loop for the second half of B.
    # The A part is the same.
    # The B part is shifted by 4 in the k-index.
    
    for a_comb, b_comb in strassen_products:
        for (a_block, a_coeff) in a_comb:
            for (b_block, b_coeff) in b_comb:
                coeff = a_coeff * b_coeff
                if coeff == 0: continue
                
                sb_terms = [
                    ((0,0), (0,0), (0,0)),
                    ((0,1), (1,0), (0,0)),
                    ((0,0), (0,1), (0,1)),
                    ((0,1), (1,1), (0,1)),
                    ((1,0), (0,0), (1,0)),
                    ((1,1), (1,0), (1,0)),
                    ((1,0), (0,1), (1,1)),
                    ((1,1), (1,1), (1,1))
                ]
                
                for (ia, ja), (jb, kb), (ic, kc) in sb_terms:
                    i = a_block[0] * 2 + ia
                    j = a_block[1] * 2 + ja
                    U[i * m + j][col] = coeff
                    
                    j = b_block[0] * 2 + jb
                    k = b_block[1] * 2 + kb + 4 # Shift for second half
                    V[j * p + k][col] = coeff
                    
                    i = a_block[0] * 2 + ic
                    k = b_block[1] * 2 + kc + 4 # Shift for second half
                    W[k * n + i][col] = coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
