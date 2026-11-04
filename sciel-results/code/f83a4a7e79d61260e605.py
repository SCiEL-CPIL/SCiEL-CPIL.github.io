import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 49 for 4x4x4 (Nested Strassen)
    # Rank 16 for 4x4x1 (Schoolbook)
    # Total Rank 65
    
    rank_1 = 49
    rank_2 = 16
    total_rank = rank_1 + rank_2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, col, coeff=1):
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        
    # --- Part 1: Nested Strassen for 4x4x4 ---
    # We will define the 7 outer Strassen terms.
    # Each outer term is a 2x2 matrix product.
    # We will compute each 2x2 matrix product using Strassen (rank 7).
    # So for each of the 7 outer terms, we add 7 inner terms.
    # Total 49 terms.
    
    # Indices for A (4x4)
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Indices for B (4x4, cols 0-3)
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    
    # Indices for C (4x4, cols 0-3)
    # C11: 0,1,4,5
    # C12: 2,3,6,7
    # C21: 8,9,12,13
    # C22: 10,11,14,15
    
    # Define the 7 outer Strassen combinations (X, Y) where X*Y is the term.
    # X and Y are 2x2 blocks (linear combos of A and B blocks).
    # We represent X and Y as 4x1 vectors of coefficients for the 4 entries of the 2x2 block.
    # But we need to map these to the original A and B indices.
    
    # Let's define the 7 outer terms as lists of (coeff, block_index) for X and Y.
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22.
    
    outer_terms = [
        # M1 = (A11+A22) * (B11+B22)
        ([(1, 0), (1, 3)], [(1, 0), (1, 3)]),
        # M2 = (A21+A22) * B11
        ([(1, 2), (1, 3)], [(1, 0)]),
        # M3 = A11 * (B12-B22)
        ([(1, 0)], [(1, 1), (-1, 3)]),
        # M4 = A22 * (B21-B11)
        ([(1, 3)], [(1, 2), (-1, 0)]),
        # M5 = (A11+A12) * B22
        ([(1, 0), (1, 1)], [(1, 3)]),
        # M6 = (A21-A11) * (B11+B12)
        ([(1, 2), (-1, 0)], [(1, 0), (1, 1)]),
        # M7 = (A12-A22) * (B21+B22)
        ([(1, 1), (-1, 3)], [(1, 2), (1, 3)])
    ]
    
    # Map block index to 4 indices in U/V/W
    A_block_indices = {
        0: [0, 1, 4, 5],   # A11
        1: [2, 3, 6, 7],   # A12
        2: [8, 9, 12, 13], # A21
        3: [10, 11, 14, 15] # A22
    }
    B_block_indices = {
        0: [0, 1, 4, 5],   # B11
        1: [2, 3, 6, 7],   # B12
        2: [8, 9, 12, 13], # B21
        3: [10, 11, 14, 15] # B22
    }
    
    # For each outer term, we have X and Y.
    # X is a 2x2 matrix. Y is a 2x2 matrix.
    # We need to compute X*Y using Strassen (rank 7).
    # Strassen for 2x2 matrices:
    # Let X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]].
    # P1 = (x00+x11)(y00+y11)
    # P2 = (x10+x11)y00
    # P3 = x00(y01-y11)
    # P4 = x11(y10-y00)
    # P5 = (x00+x01)y11
    # P6 = (x10-x00)(y00+y01)
    # P7 = (x01-x11)(y10+y11)
    # Result Z = [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    
    # Each Pi is a scalar product of two linear combos of X and Y entries.
    # X and Y entries are linear combos of A and B entries.
    # So Pi is a rank-1 term in the A,B space?
    # No, Pi is a product of two linear combos.
    # (sum a_i A_i) * (sum b_j B_j) = sum a_i b_j (A_i B_j).
    # A_i and B_j are scalars (entries of A and B).
    # So each Pi generates 4 rank-1 terms?
    # No, X and Y are 2x2 matrices.
    # The entries x00, x01, etc are linear combos of 4 A-entries each.
    # So x00 = sum c_k A_k.
    # y00 = sum d_l B_l.
    # P1 = (x00+x11)(y00+y11).
    # This is a product of two linear combos.
    # It expands to sum c_k d_l A_k B_l.
    # This is a sum of rank-1 terms.
    
    # We will iterate through the 7 outer terms.
    # For each, we determine the 4 entries of X and Y (as linear combos of A/B indices).
    # Then we apply the 7 Strassen formulas to these entries.
    # Each Strassen formula Pi is a product of two linear combos.
    # We expand each product into rank-1 terms and add to U, V, W.
    
    col = 0
    for x_combos, y_combos in outer_terms:
        # x_combos is list of (coeff, block_idx)
        # We need to form the 4 entries of X: x00, x01, x10, x11.
        # x00 is sum of coeff * A_block[0]
        # x01 is sum of coeff * A_block[1]
        # etc.
        
        x_entries = [[0]*16 for _ in range(4)] # 4 entries, each is vector of 16
        y_entries = [[0]*16 for _ in range(4)]
        
        for coeff, block_idx in x_combos:
            indices = A_block_indices[block_idx]
            # indices[0] -> x00, indices[1] -> x01, indices[2] -> x10, indices[3] -> x11
            for k in range(4):
                x_entries[k][indices[k]] += coeff
                
        for coeff, block_idx in y_combos:
            indices = B_block_indices[block_idx]
            for k in range(4):
                y_entries[k][indices[k]] += coeff
                
        # Now x_entries[k] is a vector of length 16 representing the linear combo for X entry k.
        # Same for y_entries.
        
        # Apply Strassen to X and Y.
        # P1 = (x00+x11)(y00+y11)
        # We need to add the product of two linear combos.
        # Product of (sum a_i A_i) and (sum b_j B_j) is sum a_i b_j A_i B_j.
        # This adds to U[i], V[j], W[k] for the corresponding C entry.
        
        # We need to know which C entry each P contributes to.
        # Z00 = P1+P4-P5+P7
        # Z01 = P3+P5
        # Z10 = P2+P4
        # Z11 = P1-P2+P3+P6
        
        # Z is the 2x2 result for this outer term.
        # This Z is added to the corresponding C block.
        # We need to map the 7 outer terms to C blocks.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for outer term idx, we know which C block it goes to and with what sign.
        c_block_map = {
            0: [(0, 1), (1, 1), (3, -1), (6, 1)], # M1 to C11(+), C22(+); M4 to C11(+), C21(+); etc.
            # This is getting complicated.
            # Let's just compute the 4 Z entries for the current outer term,
            # and then add them to the appropriate C block entries with the correct sign.
        }
        
        # Let's define the contribution of each outer term to C blocks.
        # outer_idx 0 (M1): C11 (+1), C22 (+1)
        # outer_idx 1 (M2): C21 (+1), C22 (-1)
        # outer_idx 2 (M3): C12 (+1), C22 (+1)
        # outer_idx 3 (M4): C11 (+1), C21 (+1)
        # outer_idx 4 (M5): C11 (-1), C12 (+1)
        # outer_idx 5 (M6): C22 (+1)
        # outer_idx 6 (M7): C11 (+1)
        
        # We will compute the 4 Z entries (Z00, Z01, Z10, Z11) for this outer term.
        # Z00 is a linear combo of A and B? No, Z00 is a scalar (entry of the 2x2 product).
        # Wait, X and Y are 2x2 matrices of linear combos.
        # The product Z is a 2x2 matrix of linear combos of products.
        # Each entry Z_uv is a sum of rank-1 terms.
        
        # We will generate the rank-1 terms for Z00, Z01, Z10, Z11.
        # And then add them to the U, V, W matrices for the corresponding C indices.
        
        # To do this, we need to know which C index Z00, etc. correspond to.
        # For a given outer term, Z is a 2x2 matrix.
        # It is added to a specific 2x2 block of C.
        # So Z00 goes to C_block[0], Z01 to C_block[1], etc.
        
        # We need a list of (c_block_idx, sign) for the current outer term.
        outer_to_c = [
            [(0, 1), (3, 1)], # M1: C11(+), C22(+)
            [(2, 1), (3, -1)], # M2: C21(+), C22(-)
            [(1, 1), (3, 1)], # M3: C12(+), C22(+)
            [(0, 1), (2, 1)], # M4: C11(+), C21(+)
            [(0, -1), (1, 1)], # M5: C11(-), C12(+)
            [(3, 1)], # M6: C22(+)
            [(0, 1)] # M7: C11(+)
        ]
        
        # Calculate Z entries using Strassen formulas on x_entries and y_entries.
        # P1 = (x00+x11)(y00+y11)
        # We need a function to add the product of two linear combos to U,V,W.
        # But we need to add it to specific W indices.
        
        # Let's compute the 7 P vectors (each is a list of (u_idx, v_idx, coeff) triples? No.)
        # A product of linear combo L_A and L_B generates terms u_i v_j w_k.
        # Here w_k is the C index.
        
        # Let's create a helper that takes two linear combos (vectors of length 16) and a target C index,
        # and adds the rank-1 terms to U, V, W.
        
        def add_product(l_a, l_b, w_idx, col_offset, sign=1):
            # l_a, l_b are vectors of length 16
            for i in range(16):
                if l_a[i] != 0:
                    for j in range(16):
                        if l_b[j] != 0:
                            coeff = sign * l_a[i] * l_b[j]
                            if coeff != 0:
                                U[i][col_offset] += coeff
                                V[j][col_offset] += coeff
                                W[w_idx][col_offset] += coeff
                                
        # This is O(16*16) per product.
        # 7 P's per outer term. 7 outer terms.
        # Total 49 products.
        # Each product adds up to 16*16 terms? No, only non-zero ones.
        # This will be slow if not optimized, but 120s is plenty.
        
        # However, we need to handle the column index correctly.
        # Each rank-1 term gets its own column?
        # No, the "rank" is the number of columns.
        # If we expand a product into multiple rank-1 terms, they all share the same column?
        # No. A rank-1 term is ONE column.
        # The product of two linear combos is a SUM of rank-1 terms.
        # So it requires MULTIPLE columns.
        
        # My previous logic was flawed.
        # If I use Strassen, I am using 7 "multiplications".
        # If each "multiplication" is a 2x2 matrix product, and I use schoolbook (rank 8) for that,
        # then each outer term uses 8 columns.
        # Total 7 * 8 = 56 columns.
        
        # If I use Strassen for the inner level, each outer term uses 7 columns.
        # Total 7 * 7 = 49 columns.
        
        # So, for each outer term, I will assign 7 columns (for the inner Strassen).
        # And for each inner Strassen term (P1..P7), I will expand it into rank-1 terms?
        # No. P1 is a product of two linear combos.
        # If I treat P1 as a single "multiplication", it is NOT a rank-1 term in the original space.
        # It is a sum of rank-1 terms.
        # So I cannot just assign 1 column to P1.
        
        # I must expand everything into rank-1 terms of the original A, B, C.
        # So the total rank is the number of scalar multiplications.
        # Nested Strassen uses 49 scalar multiplications.
        # So I need 49 columns.
        
        # I will iterate through the 49 scalar multiplications and assign a column to each.
        
        col = 0
        for outer_idx in range(7):
            x_combos, y_combos = outer_terms[outer_idx]
            
            # Form x_entries and y_entries (vectors of length 16)
            x_entries = [[0]*16 for _ in range(4)]
            y_entries = [[0]*16 for _ in range(4)]
            
            for coeff, block_idx in x_combos:
                indices = A_block_indices[block_idx]
                for k in range(4):
                    x_entries[k][indices[k]] += coeff
                    
            for coeff, block_idx in y_combos:
                indices = B_block_indices[block_idx]
                for k in range(4):
                    y_entries[k][indices[k]] += coeff
                    
            # Apply Strassen to get 7 P's (each is a pair of linear combos)
            # P1 = (x00+x11) * (y00+y11)
            # We need to add the product of (x00+x11) and (y00+y11) to the appropriate C indices.
            
            # Let's define the 7 P pairs.
            p_pairs = [
                (x_entries[0] + x_entries[2], y_entries[0] + y_entries[2]), # P1 (using list elementwise add)
                (x_entries[2] + x_entries[3], y_entries[0]), # P2
                (x_entries[0], y_entries[1] - y_entries[3]), # P3
                (x_entries[3], y_entries[2] - y_entries[0]), # P4
                (x_entries[0] + x_entries[1], y_entries[3]), # P5
                (x_entries[2] - x_entries[0], y_entries[0] + y_entries[1]), # P6
                (x_entries[1] - x_entries[3], y_entries[2] + y_entries[3]) # P7
            ]
            
            # Helper to add elementwise sum/diff of lists
            def add_lists(l1, l2, sign=1):
                return [l1[i] + sign * l2[i] for i in range(16)]
                
            # Recalculate p_pairs with helper
            p_pairs = [
                (add_lists(x_entries[0], x_entries[2]), add_lists(y_entries[0], y_entries[2])),
                (add_lists(x_entries[2], x_entries[3]), y_entries[0]),
                (x_entries[0], add_lists(y_entries[1], y_entries[3], -1)),
                (x_entries[3], add_lists(y_entries[2], y_entries[0], -1)),
                (add_lists(x_entries[0], x_entries[1]), y_entries[3]),
                (add_lists(x_entries[2], x_entries[0], -1), add_lists(y_entries[0], y_entries[1])),
                (add_lists(x_entries[1], x_entries[3], -1), add_lists(y_entries[2], y_entries[3]))
            ]
            
            # Now, for each P pair, we expand into rank-1 terms.
            # P_pair is (l_a, l_b).
            # We add terms to U, V, W for the C indices determined by outer_to_c.
            
            c_targets = outer_to_c[outer_idx]
            
            # For each P in the 7 inner Strassen terms, we need to know which Z entry it contributes to.
            # Z00 = P1+P4-P5+P7
            # Z01 = P3+P5
            # Z10 = P2+P4
            # Z11 = P1-P2+P3+P6
            
            # So for each P_idx (0..6), we have a list of (z_entry_idx, sign).
            p_to_z = [
                [(0, 1), (3, 1)], # P1
                [(2, 1), (3, -1)], # P2
                [(1, 1), (3, 1)], # P3
                [(0, 1), (2, 1)], # P4
                [(0, -1), (1, 1)], # P5
                [(3, 1)], # P6
                [(0, 1)] # P7
            ]
            
            for p_idx in range(7):
                l_a, l_b = p_pairs[p_idx]
                # Expand l_a * l_b into rank-1 terms
                for i in range(16):
                    if l_a[i] != 0:
                        for j in range(16):
                            if l_b[j] != 0:
                                coeff_base = l_a[i] * l_b[j]
                                if coeff_base == 0: continue
                                
                                # This rank-1 term contributes to multiple Z entries?
                                # No, this specific scalar multiplication (A_i * B_j) is used in the calculation of P.
                                # P contributes to multiple Z entries.
                                # So this scalar multiplication contributes to multiple C entries.
                                # We need to add to W for each target C entry.
                                
                                # But U and V are fixed for this scalar multiplication.
                                # W changes.
                                # Can one column have multiple W entries?
                                # No. A rank-1 term is u v w.
                                # If it contributes to C1 and C2, it must be two separate rank-1 terms?
                                # Or can w have multiple non-zeros?
                                # The tensor decomposition allows w to be any vector.
                                # So we can have one column where W has non-zeros at multiple indices.
                                
                                # So we just add to W for all targets.
                                
                                for z_idx, sign in p_to_z[p_idx]:
                                    # z_idx is 0..3 corresponding to Z00..Z11.
                                    # We need to map z_idx to C block indices.
                                    # For this outer term, Z is added to specific C blocks.
                                    # c_targets is list of (c_block_idx, sign_outer).
                                    # This is getting too nested.
                                    
                                    # Let's simplify:
                                    # Just add the term to a new column.
                                    # If it contributes to multiple C's, we set W for all of them.
                                    
                                    # But we are inside the loop for i, j.
                                    # We should create ONE column for this (i, j) pair?
                                    # No, the rank is the number of columns.
                                    # If we create a column for every non-zero (i, j) pair, the rank is huge.
                                    
                                    # The rank is the number of SCALAR MULTIPLICATIONS.
                                    # In Strassen, we have 49 scalar multiplications.
                                    # Each scalar multiplication is of the form (linear_combo_A) * (linear_combo_B).
                                    # This is NOT a single rank-1 term in the original basis.
                                    # It is a sum of rank-1 terms.
                                    
                                    # So I CANNOT use Strassen directly without expanding.
                                    # Expanding increases the rank.
                                    # The rank of the expansion is the number of non-zero terms in the product of the linear combos.
                                    
                                    # This is why Rank 49 is an upper bound, but the actual rank might be higher if we don't merge terms.
                                    # However, if we are careful, the rank is exactly 49.
                                    # How?
                                    # Because the "scalar multiplications" in Strassen are the rank-1 terms.
                                    # The "linear combos" are the coefficients of U and V.
                                    # So, for each of the 49 Strassen steps, we have ONE column.
                                    # U has the coefficients of the linear combo of A.
                                    # V has the coefficients of the linear combo of B.
                                    # W has the coefficients of the linear combo of C.
                                    
                                    # So, I should NOT expand into i, j loops.
                                    # I should just assign the linear combos directly to U and V.
                                    
                                    # Let's restart the logic for Part 1.
                                    pass
