import numpy as np
import time

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Dimensions
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Rank components:
    # 1. Core 4x4x4 (j in 0..3, k in 0..3): Rank 28
    # 2. Strip 4x4x2 (j in 0..3, k in 4..5): Rank 2
    # 3. Strip 4x1x6 (j=4, k in 0..5): Rank 1
    
    total_rank = 28 + 2 + 1
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: Core 4x4x4 using Strassen (Rank 28) ---
    # We implement the 7 Strassen products, each expanded into 4 schoolbook terms.
    # Strassen products M1..M7 are 2x2 matrix products.
    # Each 2x2 product P = A_block @ B_block is decomposed into 4 terms:
    # P[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] ... wait, schoolbook for 2x2 is rank 4.
    # Terms:
    # (A00 B00, A01 B10, A10 B01, A11 B11)
    
    # Strassen definitions for 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
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
    
    # We need to map these to U, V, W indices.
    # The core corresponds to i in 0..3, j in 0..3, k in 0..3.
    # Block indices:
    # i_block = i // 2, j_block = j // 2, k_block = k // 2
    # i_local = i % 2, j_local = j % 2, k_local = k % 2
    
    # We will iterate through the 7 Strassen terms.
    # For each term, we have a linear combination of A blocks and B blocks.
    # Then we expand the 2x2 product into 4 rank-1 terms.
    
    # Helper to add a rank-1 term for a specific 2x2 block product
    # A_block is 2x2, B_block is 2x2.
    # The product contributes to C_block.
    # We need to add 4 terms to U, V, W.
    
    # Term index offset for the core part
    col = 0
    
    # Define the 7 Strassen terms as lists of (A_coeff, B_coeff, C_coeff)
    # Actually, it's easier to just generate the U, V, W vectors directly.
    
    # For each Strassen product M_k, we have:
    # M_k = A_comb @ B_comb
    # Where A_comb and B_comb are 2x2 matrices with coefficients.
    
    # Let's define the coefficients for A and B blocks for each M_k.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    
    # M1: A = A00+A11, B = B00+B11
    # M2: A = A10+A11, B = B00
    # M3: A = A00, B = B01-B11
    # M4: A = A11, B = B10-B00
    # M5: A = A00+A01, B = B11
    # M6: A = A10-A00, B = B00+B01
    # M7: A = A01-A11, B = B10+B11
    
    # For each M_k, we expand A_comb @ B_comb into 4 terms.
    # A_comb[i, j] * B_comb[j, k]
    
    # We need to map the block indices to global indices.
    # Global i = 2*i_block + i_local
    # Global j = 2*j_block + j_local
    # Global k = 2*k_block + k_local
    
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # We will iterate through the 7 terms.
    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict of (block_row, block_col) -> coeff
        # B_coeffs: dict of (block_row, block_col) -> coeff
        # C_coeffs: dict of (block_row, block_col) -> coeff (for output)
        # Actually, we just need to add the rank-1 terms for the product.
        # The output C is reconstructed by summing the contributions.
        # But we are building the tensor decomposition, so we just add the rank-1 terms
        # corresponding to the multiplication A_comb @ B_comb.
        # The coefficients in U and V will be the coefficients from A_comb and B_comb.
        # The coefficient in W will be 1 (or the coefficient from C if we were doing a specific C, 
        # but here we are decomposing the tensor, so W is just 1 for the matching indices).
        
        # Wait, the tensor entry is 1.
        # If A_comb has coeff 2, then U will have 2.
        # If B_comb has coeff -1, then V will have -1.
        # Then U*V*W = 2 * (-1) * 1 = -2.
        # This correctly contributes to the tensor?
        # The tensor is linear in A and B.
        # Yes, this is how bilinear decompositions work.
    ]
    
    # Let's define the terms explicitly.
    # Each term is a list of (a_row, a_col, b_row, b_col, coeff)
    # Where A_block[a_row, a_col] and B_block[b_row, b_col] are multiplied.
    
    # M1 = (A00+A11)(B00+B11)
    # A00*B00, A00*B11, A11*B00, A11*B11
    # M2 = (A10+A11)B00
    # A10*B00, A11*B00
    # M3 = A00(B01-B11)
    # A00*B01, -A00*B11
    # M4 = A11(B10-B00)
    # A11*B10, -A11*B00
    # M5 = (A00+A01)B11
    # A00*B11, A01*B11
    # M6 = (A10-A00)(B00+B01)
    # A10*B00, A10*B01, -A00*B00, -A00*B01
    # M7 = (A01-A11)(B10+B11)
    # A01*B10, A01*B11, -A11*B10, -A11*B11
    
    # Total 4+2+2+2+2+4+4 = 20 terms?
    # No, each 2x2 product has 4 terms.
    # M1: 4 terms.
    # M2: 2 terms? No, (A10+A11) is a 2x2 matrix. B00 is 2x2.
    # Product is 2x2. So 4 terms.
    # All M_k are 2x2 products, so each has 4 terms.
    # Total 7 * 4 = 28 terms.
    
    # We need to map the block indices to local 2x2 indices.
    # A_block[ar, ac] corresponds to A[2*ar+lr, 2*ac+lc]
    # B_block[br, bc] corresponds to B[2*br+lr, 2*bc+lc]
    # Wait, the product is A_block @ B_block.
    # (A_block @ B_block)[lr, lk] = sum_lc A_block[lr, lc] B_block[lc, lk]
    
    # So for each M_k, we iterate lr, lc, lk.
    # Term: A_block[lr, lc] * B_block[lc, lk]
    
    # Let's define the coefficients for A and B blocks for each M_k.
    # A_coeffs[ar, ac]
    # B_coeffs[br, bc]
    
    M1_A = {(0,0): 1, (1,1): 1}
    M1_B = {(0,0): 1, (1,1): 1}
    
    M2_A = {(1,0): 1, (1,1): 1}
    M2_B = {(0,0): 1}
    
    M3_A = {(0,0): 1}
    M3_B = {(0,1): 1, (1,1): -1}
    
    M4_A = {(1,1): 1}
    M4_B = {(1,0): 1, (0,0): -1}
    
    M5_A = {(0,0): 1, (0,1): 1}
    M5_B = {(1,1): 1}
    
    M6_A = {(1,0): 1, (0,0): -1}
    M6_B = {(0,0): 1, (0,1): 1}
    
    M7_A = {(0,1): 1, (1,1): -1}
    M7_B = {(1,0): 1, (1,1): 1}
    
    strassen_data = [
        (M1_A, M1_B),
        (M2_A, M2_B),
        (M3_A, M3_B),
        (M4_A, M4_B),
        (M5_A, M5_B),
        (M6_A, M6_B),
        (M7_A, M7_B)
    ]
    
    for A_coeffs, B_coeffs in strassen_data:
        # Iterate over all possible local indices
        for lr in range(2):
            for lc in range(2):
                for lk in range(2):
                    # Sum over lc? No, the term is A[lr, lc] * B[lc, lk]
                    # But A_coeffs and B_coeffs are for the blocks.
                    # We need to multiply the coefficients.
                    # Actually, the product is sum_lc A[lr, lc] B[lc, lk].
                    # So we need to iterate lc as well.
                    pass # The loop structure is wrong.
                    
    # Correct loop:
    col = 0
    for A_coeffs, B_coeffs in strassen_data:
        for lr in range(2):
            for lk in range(2):
                for lc in range(2):
                    a_coeff = A_coeffs.get((lr, lc), 0)
                    b_coeff = B_coeffs.get((lc, lk), 0)
                    if a_coeff == 0 or b_coeff == 0:
                        continue
                    # This term contributes to C[lr, lk]
                    # Global indices:
                    # i = 2*0 + lr? No, block row/col.
                    # The Strassen blocks are 2x2.
                    # A_block[ar, ac] is at global rows 2*ar, 2*ar+1 and cols 2*ac, 2*ac+1.
                    # Wait, the block indices ar, ac are 0..1.
                    # Local indices lr, lc, lk are 0..1.
                    # Global i = 2*ar + lr
                    # Global j = 2*ac + lc
                    # Global k = 2*bc + lk? No, B_block[br, bc].
                    # B_block[br, bc] is at global rows 2*br, 2*br+1 and cols 2*bc, 2*bc+1.
                    # In the product A_block @ B_block, the row index of B is lc.
                    # So br = lc.
                    # The col index of B is lk.
                    # So bc = lk.
                    # But wait, A_block[ar, ac] and B_block[br, bc] must match in the middle.
                    # In Strassen, the blocks are aligned.
                    # So ar, ac, br, bc are the block coordinates.
                    # The local indices lr, lc, lk are within the 2x2 block.
                    
                    # For a specific term in M_k:
                    # It involves A_block[ar, ac] and B_block[br, bc].
                    # The product is A_block[ar, ac] @ B_block[br, bc].
                    # This requires ac = br? No, they are just block labels.
                    # In Strassen, the blocks are the same size and aligned.
                    # So the "middle" index is the block column of A and block row of B.
                    # So we must have ac = br.
                    
                    # Let's re-evaluate.
                    # M1 = (A00+A11)(B00+B11)
                    # A00 is block (0,0). B00 is block (0,0).
                    # Product A00 @ B00.
                    # A01 is block (0,1). B11 is block (1,1).
                    # Product A01 @ B11? No, A01 @ B01?
                    # Strassen:
                    # M1 = (A00+A11)(B00+B11)
                    # This expands to A00B00 + A00B11 + A11B00 + A11B11.
                    # A00B11: A block (0,0) @ B block (1,1).
                    # Is this a valid matrix product?
                    # A00 is 2x2. B11 is 2x2.
                    # Yes, they are both 2x2 matrices.
                    # So we can multiply them.
                    
                    # So for each pair (A_block, B_block) in the expansion:
                    # We add 4 terms (for lr, lc, lk).
                    
    # Let's restart the loop logic.
    
    col = 0
    for A_coeffs, B_coeffs in strassen_data:
        # Iterate over all non-zero A blocks
        for (ar, ac), a_val in A_coeffs.items():
            # Iterate over all non-zero B blocks
            for (br, bc), b_val in B_coeffs.items():
                # The product is A_block[ar, ac] @ B_block[br, bc]
                # This is valid.
                # Expand into 4 rank-1 terms
                for lr in range(2):
                    for lc in range(2):
                        for lk in range(2):
                            # Term: A[2*ar+lr, 2*ac+lc] * B[2*br+lc, 2*bc+lk]
                            # Wait, the middle index is lc.
                            # A col is lc. B row is lc.
                            # So yes.
                            
                            # Global indices:
                            i = 2 * ar + lr
                            j = 2 * ac + lc
                            k = 2 * bc + lk
                            
                            # Note: br must be lc? No.
                            # In the product A_block @ B_block, the column of A is lc and row of B is lc.
                            # But A_block and B_block are just 2x2 matrices.
                            # The "block" indices ar, ac, br, bc are just labels.
                            # The multiplication is just 2x2 @ 2x2.
                            # So the global j index is 2*ac + lc.
                            # And the global j index from B is 2*br + lc.
                            # For the tensor to be consistent, these must be the same j.
                            # So we require 2*ac + lc == 2*br + lc => ac == br.
                            # If ac != br, the term does not contribute to the tensor?
                            # No, the tensor is defined for all i, j, k.
                            # If we multiply A[2*ar+lr, 2*ac+lc] and B[2*br+lc, 2*bc+lk],
                            # the j index is 2*ac+lc.
                            # But B is at row 2*br+lc.
                            # If ac != br, then the j index in A and B are different.
                            # This term would correspond to A[i, j1] B[j2, k] with j1 != j2.
                            # This is 0 in the multiplication tensor.
                            # So we should only add the term if ac == br.
                            
                            if ac != br:
                                continue
                                
                            # Add the term
                            U[i * m + j, col] = a_val
                            V[j * p + k, col] = b_val
                            W[k * n + i, col] = 1
                            col += 1
                            
    # col should be 28.
    
    # --- Part 2: Strip 4x4x2 (j in 0..3, k in 4..5) ---
    # Rank 2.
    # Term 1: k=4
    U[0:20, col] = 1 # i in 0..3, j in 0..3. All 1s.
    # Wait, U index is i*m + j.
    # For i in 0..3, j in 0..3:
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
    for j in range(4):
        V[j * p + 4, col] = 1
    for i in range(4):
        W[4 * n + i, col] = 1
    col += 1
    
    # Term 2: k=5
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
    for j in range(4):
        V[j * p + 5, col] = 1
    for i in range(4):
        W[5 * n + i, col] = 1
    col += 1
    
    # --- Part 3: Strip 4x1x6 (j=4, k in 0..5) ---
    # Rank 1.
    for i in range(4):
        U[i * m + 4, col] = 1
    for k in range(6):
        V[4 * p + k, col] = 1
    for k in range(6):
        for i in range(4):
            W[k * n + i, col] = 1
    col += 1
    
    # col should be 31.
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
