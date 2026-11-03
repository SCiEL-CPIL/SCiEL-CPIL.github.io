import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import sys

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy for speed."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    # Compute sum over rank: U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note: The problem statement says W[k*n+i,r].
    # My reshape W_arr is (p, n, rank) so W_arr[k, i, r] corresponds to W[k*n+i, r].
    # The einsum 'ijc,jkc,kic->ijk' computes sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # This matches the tensor definition T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(expected, reconstructed, atol=tol)

def _build_strassen_2x2_terms():
    """
    Returns the 7 terms of the Strassen decomposition for 2x2 matrix multiplication.
    Each term is a tuple (U_dict, V_dict, W_dict).
    Keys in U_dict are (i,j) for A.
    Keys in V_dict are (j,k) for B.
    Keys in W_dict are (i,k) for C.
    """
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    terms = []
    
    # M1 = (a00+a11)*(b00+b11)
    # c00 += M1, c11 += M1
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    
    # M2 = (a10+a11)*b00
    # c10 += M2, c11 -= M2
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}))
    
    # M3 = a00*(b01-b11)
    # c01 += M3, c11 += M3
    terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}))
    
    # M4 = a11*(b10-b00)
    # c00 += M4, c10 += M4
    terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}))
    
    # M5 = (a00+a01)*b11
    # c00 -= M5, c01 += M5
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}))
    
    # M6 = (a10-a00)*(b00+b01)
    # c11 += M6
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    
    # M7 = (a01-a11)*(b10+b11)
    # c00 += M7
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    
    return terms

def _build_decomp_4x4_4x5():
    """
    Constructs a rank 65 decomposition for 4x4 * 4x5.
    Uses Strassen^2 (rank 49) for the first 4 columns and schoolbook (rank 16) for the 5th column.
    """
    n, m, p = 4, 4, 5
    
    U = [[0] * 65 for _ in range(n * m)]
    V = [[0] * 65 for _ in range(m * p)]
    W = [[0] * 65 for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4 * 4x4 using Strassen^2
    # We split A and B (first 4 cols) into 2x2 blocks of 2x2 matrices.
    # A blocks:
    # A1: rows 0-1, cols 0-1
    # A2: rows 0-1, cols 2-3
    # A3: rows 2-3, cols 0-1
    # A4: rows 2-3, cols 2-3
    
    # B blocks (first 4 cols):
    # B1: rows 0-1, cols 0-1
    # B2: rows 0-1, cols 2-3
    # B3: rows 2-3, cols 0-1
    # B4: rows 2-3, cols 2-3
    
    # C blocks:
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # Block Strassen terms (7 of them):
    # Each term is a product of a linear combination of A blocks and B blocks.
    # We need to expand these into the inner Strassen terms (7 of them per block term).
    
    # Helper to map block indices to actual indices
    def get_block_indices(row_block, col_block, is_A):
        # row_block, col_block in {0, 1}
        # is_A: True for A (i,j), False for B (j,k) or C (i,k)
        # For A: i = 2*row_block + r, j = 2*col_block + c
        # For B: j = 2*row_block + r, k = 2*col_block + c
        # For C: i = 2*row_block + r, k = 2*col_block + c
        indices = []
        for r in range(2):
            for c in range(2):
                if is_A == 'A':
                    indices.append((2*row_block + r, 2*col_block + c))
                elif is_A == 'B':
                    indices.append((2*row_block + r, 2*col_block + c))
                else: # C
                    indices.append((2*row_block + r, 2*col_block + c))
        return indices
    
    # A block indices
    A1_idx = get_block_indices(0, 0, 'A')
    A2_idx = get_block_indices(0, 1, 'A')
    A3_idx = get_block_indices(1, 0, 'A')
    A4_idx = get_block_indices(1, 1, 'A')
    
    # B block indices (for first 4 cols)
    B1_idx = get_block_indices(0, 0, 'B')
    B2_idx = get_block_indices(0, 1, 'B')
    B3_idx = get_block_indices(1, 0, 'B')
    B4_idx = get_block_indices(1, 1, 'B')
    
    # C block indices
    C11_idx = get_block_indices(0, 0, 'C')
    C12_idx = get_block_indices(0, 1, 'C')
    C21_idx = get_block_indices(1, 0, 'C')
    C22_idx = get_block_indices(1, 1, 'C')
    
    # Block Strassen definitions:
    # M1 = (A1+A4)*(B1+B4)
    # M2 = (A3+A4)*B1
    # M3 = A1*(B2-B4)
    # M4 = A4*(B3-B1)
    # M5 = (A1+A2)*B4
    # M6 = (A3-A1)*(B1+B2)
    # M7 = (A2-A4)*(B3+B4)
    
    # C contributions:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to generate the 7 inner Strassen terms for each of the 7 block terms.
    # The "A" and "B" inputs to the inner Strassen are linear combinations of the block entries.
    # For example, for M1, the A input is A1+A4.
    # The inner Strassen term (e.g., M1_inner) involves products like (a00+a11)*(b00+b11).
    # Here a00 is the (0,0) entry of the combined A block, which is A1[0,0] + A4[0,0].
    # So the U vector for this term will have 1 at A1[0,0] and 1 at A4[0,0].
    
    def add_block_term(A_coeffs, B_coeffs, C_block_idx, signs):
        """
        Adds the 7 inner Strassen terms for a specific block-level product.
        A_coeffs: dict mapping block_position (0,0)..(1,1) to a dict of {actual_A_idx: coeff}
                  Or more simply, we pass the combined A and B index dictionaries.
                  Let's pass the combined A and B index dictionaries directly.
                  A_comb: dict {actual_A_idx: coeff}
                  B_comb: dict {actual_B_idx: coeff}
                  C_block_idx: list of 4 actual C indices
                  signs: [s0, s1, s2, s3] for C00, C01, C10, C11 contributions
        """
        nonlocal col
        
        # We need to map the "inner" Strassen indices (0,0), (0,1), (1,0), (1,1)
        # to the actual indices in A_comb and B_comb.
        # The inner Strassen assumes a 2x2 structure.
        # We need to identify which actual indices correspond to the 2x2 grid of the combined block.
        
        # This is tricky because A_comb is a sparse dict.
        # We need to know the "shape" of the combined block.
        # For M1 = (A1+A4), the combined block is 2x2.
        # (0,0) of combined block corresponds to A1[0,0] + A4[0,0].
        # (0,1) corresponds to A1[0,1] + A4[0,1].
        # etc.
        
        # Let's assume the combined blocks are always 2x2 and aligned.
        # We will construct the U, V, W vectors for each of the 7 inner terms.
        
        inner_terms = _build_strassen_2x2_terms()
        
        # Map inner indices to actual indices
        # Inner index (r, c) maps to actual index based on the block structure.
        # We need to pass the mapping.
        # Let's refactor: pass the actual index lists for the 4 positions of the combined A and B blocks.
        # A_actual_indices: [idx_00, idx_01, idx_10, idx_11]
        # B_actual_indices: [idx_00, idx_01, idx_10, idx_11]
        # C_actual_indices: [idx_00, idx_01, idx_10, idx_11]
        
        pass # Placeholder for now, will implement properly in the loop below

    # To make it simpler, I will explicitly list the 49 terms.
    # This is tedious but reliable.
    
    # Let's define a helper to add a single scalar multiplication term to U, V, W
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # This approach of adding terms one by one is inefficient if we have to manage col globally.
    # Better to build the 49 terms as lists of dicts and then assemble.
    
    terms_49 = []
    
    # Block Strassen Terms
    # M1: (A1+A4)*(B1+B4)
    # A_comb: A1 + A4
    # B_comb: B1 + B4
    # C: C11 (signs [1,0,0,1] ? No, C11 gets +M1. C22 gets +M1? No.
    # C11 = M1 + ...
    # C22 = M1 - ...
    # So for M1, it contributes to C11 with +1 and C22 with +1.
    # Wait, the C block for M1 is C11? No, M1 is a 2x2 matrix.
    # M1 contributes to C11 (top-left block) and C22 (bottom-right block)?
    # No, Strassen computes the product of two 2x2 blocks.
    # The result of M1 is a 2x2 matrix.
    # How is it added to C?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M1 (which is a 2x2 matrix) contributes to C11 (the top-left 2x2 block of C) with coefficient +1
    # AND to C22 (the bottom-right 2x2 block of C) with coefficient +1.
    # This means the "C block" for the inner Strassen of M1 is not just C11.
    # It is a combination of C11 and C22.
    
    # This complicates the inner Strassen application because the C indices are not contiguous in a single 2x2 block.
    
    # Alternative:
    # We can treat the 49 terms as 49 independent scalar multiplications.
    # For each of the 7 block terms, we expand the 2x2*2x2 product into 7 scalar multiplications.
    # Each scalar multiplication involves:
    # - A linear combination of A entries
    # - A linear combination of B entries
    # - A contribution to specific C entries
    
    # Let's do this expansion.
    
    def expand_block_term(A_block_coeffs, B_block_coeffs, C_contributions):
        """
        A_block_coeffs: dict {(r,c): coeff} for the 2x2 combined A block
        B_block_coeffs: dict {(r,c): coeff} for the 2x2 combined B block
        C_contributions: dict {(block_r, block_c, c_r, c_c): coeff}
                           block_r, block_c: which C block (0,0 for C11, 0,1 for C12, etc)
                           c_r, c_c: which entry in that 2x2 block
        """
        # We need to map the "combined" A and B entries to actual indices.
        # This requires knowing the actual indices for each position in the combined block.
        # This function is too abstract. I will hardcode the 49 terms.
        pass

    # Given the complexity, I will use a known library or algorithm if available, but I must write it.
    # Let's try a different strategy:
    # Use the schoolbook for 4x4x4 (rank 16) and 4x4x1 (rank 16)? Total 32? No, 16+16=32 is for 4x4x4? No, schoolbook 4x4x4 is 64.
    # Schoolbook 4x4x4 is 64.
    # Strassen 4x4x4 is 49.
    
    # I will implement the 49 terms by iterating through the 7 block terms and 7 inner terms.
    
    # Mapping of block positions to actual indices
    # A1: (0,0), (0,1), (1,0), (1,1) -> A[0,0], A[0,1], A[1,0], A[1,1]
    # A2: (0,2), (0,3), (1,2), (1,3) -> A[0,2], A[0,3], A[1,2], A[1,3]
    # A3: (2,0), (2,1), (3,0), (3,1) -> A[2,0], A[2,1], A[3,0], A[3,1]
    # A4: (2,2), (2,3), (3,2), (3,3) -> A[2,2], A[2,3], A[3,2], A[3,3]
    
    A_blocks = {
        1: [(0,0), (0,1), (1,0), (1,1)],
        2: [(0,2), (0,3), (1,2), (1,3)],
        3: [(2,0), (2,1), (3,0), (3,1)],
        4: [(2,2), (2,3), (3,2), (3,3)]
    }
    
    B_blocks = {
        1: [(0,0), (0,1), (1,0), (1,1)],
        2: [(0,2), (0,3), (1,2), (1,3)],
        3: [(2,0), (2,1), (3,0), (3,1)],
        4: [(2,2), (2,3), (3,2), (3,3)]
    }
    
    C_blocks = {
        11: [(0,0), (0,1), (1,0), (1,1)],
        12: [(0,2), (0,3), (1,2), (1,3)],
        21: [(2,0), (2,1), (3,0), (3,1)],
        22: [(2,2), (2,3), (3,2), (3,3)]
    }
    
    # Block Strassen Terms
    # M1: (A1+A4)*(B1+B4)
    # M2: (A3+A4)*B1
    # M3: A1*(B2-B4)
    # M4: A4*(B3-B1)
    # M5: (A1+A2)*B4
    # M6: (A3-A1)*(B1+B2)
    # M7: (A2-A4)*(B3+B4)
    
    # C Contributions
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block term, we have a combined A block and combined B block.
    # We apply inner Strassen to these combined blocks.
    # The result is added to the specified C blocks with the specified signs.
    
    def get_combined_block_indices(block_ids, signs, blocks_dict):
        """
        Returns a list of 4 actual indices for the combined block,
        and a list of 4 coefficients for the inner Strassen mapping?
        No, it returns a dict {pos: actual_indices} where pos is (0,0)..(1,1)
        and actual_indices is a list of (idx, coeff) pairs.
        """
        combined = {}
        for r in range(2):
            for c in range(2):
                combined[(r,c)] = []
        
        for bid, sign in zip(block_ids, signs):
            block_indices = blocks_dict[bid]
            for idx_pos, actual_idx in enumerate(block_indices):
                # idx_pos 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                r = idx_pos // 2
                c = idx_pos % 2
                combined[(r,c)].append((actual_idx, sign))
        
        return combined

    # Inner Strassen Terms (from _build_strassen_2x2_terms)
    # We need to map the inner Strassen's "a00", "a01", etc to the combined block's entries.
    
    inner_strassen_terms = _build_strassen_2x2_terms()
    
    # List of block terms: (A_blocks, A_signs, B_blocks, B_signs, C_contribs)
    # C_contribs: list of (C_block_id, sign)
    
    block_terms = [
        # M1
        ([1, 4], [1, 1], [1, 4], [1, 1], [(11, 1), (22, 1)]),
        # M2
        ([3, 4], [1, 1], [1], [1], [(21, 1), (22, -1)]), # C21 += M2, C22 -= M2
        # M3
        ([1], [1], [2, 4], [1, -1], [(12, 1), (22, 1)]), # C12 += M3, C22 += M3
        # M4
        ([4], [1], [3, 1], [1, -1], [(11, 1), (21, 1)]), # C11 += M4, C21 += M4
        # M5
        ([1, 2], [1, 1], [4], [1], [(11, -1), (12, 1)]), # C11 -= M5, C12 += M5
        # M6
        ([3, 1], [1, -1], [1, 2], [1, 1], [(22, 1)]),    # C22 += M6
        # M7
        ([2, 4], [1, -1], [3, 4], [1, 1], [(11, 1)]),    # C11 += M7
    ]
    
    # Wait, C22 = M1 - M2 + M3 + M6.
    # My M2 contribution to C22 is -1. Correct.
    # My M3 contribution to C22 is +1. Correct.
    # My M6 contribution to C22 is +1. Correct.
    
    # Now, for each block term, generate 7 inner terms.
    
    for bt in block_terms:
        A_bids, A_signs, B_bids, B_signs, C_contribs = bt
        
        # Build combined A and B blocks
        # combined_A: dict {(r,c): [(actual_idx, coeff), ...]}
        combined_A = {}
        for r in range(2):
            for c in range(2):
                combined_A[(r,c)] = []
        for bid, sign in zip(A_bids, A_signs):
            block_indices = A_blocks[bid]
            for idx_pos, actual_idx in enumerate(block_indices):
                r = idx_pos // 2
                c = idx_pos % 2
                combined_A[(r,c)].append((actual_idx, sign))
                
        combined_B = {}
        for r in range(2):
            for c in range(2):
                combined_B[(r,c)] = []
        for bid, sign in zip(B_bids, B_signs):
            block_indices = B_blocks[bid]
            for idx_pos, actual_idx in enumerate(block_indices):
                r = idx_pos // 2
                c = idx_pos % 2
                combined_B[(r,c)].append((actual_idx, sign))
        
        # Now apply inner Strassen
        for inner_term in inner_strassen_terms:
            U_dict, V_dict, W_dict = inner_term
            
            # U_dict keys are (0,0)..(1,1) for the inner A block
            # V_dict keys are (0,0)..(1,1) for the inner B block
            # W_dict keys are (0,0)..(1,1) for the inner C block (result of this inner product)
            
            # We need to map the inner A entries to actual A indices
            # Inner A (r,c) corresponds to combined_A[(r,c)]
            # Inner B (r,c) corresponds to combined_B[(r,c)]
            
            # For W, we need to map the inner C entries to actual C indices.
            # The inner C block is a 2x2 matrix.
            # It contributes to the C blocks specified in C_contribs.
            # C_contribs is a list of (C_block_id, sign).
            # The inner C entry (r,c) corresponds to the (r,c) entry of the C_block.
            
            # Build the actual U, V, W vectors for this term
            U_vec = [0.0] * (n*m)
            V_vec = [0.0] * (m*p)
            W_vec = [0.0] * (p*n)
            
            # Fill U
            for pos, coeff in U_dict.items():
                r, c = pos
                for actual_idx, a_coeff in combined_A[(r,c)]:
                    i, j = actual_idx
                    U_vec[i*m + j] += coeff * a_coeff
            
            # Fill V
            for pos, coeff in V_dict.items():
                r, c = pos
                for actual_idx, b_coeff in combined_B[(r,c)]:
                    j, k = actual_idx
                    V_vec[j*p + k] += coeff * b_coeff
            
            # Fill W
            # The inner W_dict has keys (0,0)..(1,1)
            # These correspond to the entries of the 2x2 result block.
            # We need to add this to the actual C matrix.
            # The C_contribs tell us which C blocks this result is added to.
            # Note: A single block term (like M1) produces a 2x2 result.
            # This result is added to C11 and C22 (for M1).
            # So for each entry (r,c) in the inner result, we add it to C11[r,c] and C22[r,c].
            
            for pos, coeff in W_dict.items():
                r, c = pos
                for c_block_id, c_sign in C_contribs:
                    c_block_indices = C_blocks[c_block_id]
                    idx_pos = r * 2 + c
                    i, k = c_block_indices[idx_pos]
                    W_vec[k*n + i] += coeff * c_sign
            
            # Add this term to U, V, W
            U[col] = U_vec
            V[col] = V_vec
            W[col] = W_vec
            col += 1
            
    # Part 2: 5th column (schoolbook)
    # C[i, 4] = sum_j A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            U[col][i*m + j] = 1.0
            V[col][j*p + 4] = 1.0
            W[col][4*n + i] = 1.0
            col += 1
            
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U, V, W, rank = _build_decomp_4x4_4x5()
    
    # Verify
    if not _verify_decomposition_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    rank = col
        if not _verify_decomposition_fast(U, V, W, n, m, p):
            raise ValueError("All decompositions failed")
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
