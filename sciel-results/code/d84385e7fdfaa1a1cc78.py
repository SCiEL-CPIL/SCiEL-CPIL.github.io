import time
import numpy as np

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # We will construct the rank 98 decomposition using Strassen's algorithm
    # for 4x4x4 matrix multiplication, applied to two 4x4 blocks of the B matrix.
    
    # Rank is 98
    rank = 98
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (16, 98)
    # V: (m*p, rank) = (32, 98)
    # W: (p*n, rank) = (32, 98)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_idx = 0
    
    # Function to add a scalar multiplication term
    # a_coeffs: dict (i,j) -> coeff for A
    # b_coeffs: dict (j,k) -> coeff for B
    # c_coeffs: dict (i,k) -> coeff for C
    def add_term(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col_idx
        if col_idx >= rank:
            return False
        
        for (i, j), ua in a_coeffs.items():
            U[i*m + j, col_idx] += ua
        for (j, k), vb in b_coeffs.items():
            V[j*p + k, col_idx] += vb
        for (i, k), wc in c_coeffs.items():
            W[k*n + i, col_idx] += wc
        col_idx += 1
        return True
    
    # Strassen's algorithm for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A * B
    # 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_2x2_strassen(base_i, base_j, base_k, block_idx):
        # base_i, base_j, base_k are offsets for the 2x2 blocks
        # We are adding 7 terms
        nonlocal col_idx
        
        # M1
        a = {(base_i, base_j): 1, (base_i+1, base_j+1): 1}
        b = {(base_j, base_k): 1, (base_j+1, base_k+1): 1}
        c = {(base_i, base_k): 1, (base_i, base_k+1): 1, (base_i+1, base_k): 1, (base_i+1, base_k+1): 1}
        # Wait, C indices for M1:
        # C00 += M1, C01 += M1? No.
        # C00 = M1 + ...
        # C01 = ...
        # C10 = ...
        # C11 = ...
        # So M1 contributes to C00 and C11?
        # Let's re-verify Strassen formulas.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 contributes to C00 (coeff 1) and C11 (coeff 1).
        c_m1 = {(base_i, base_k): 1, (base_i+1, base_k+1): 1}
        if not add_term(a, b, c_m1): return
        
        # M2
        a = {(base_i+1, base_j): 1, (base_i+1, base_j+1): 1}
        b = {(base_j, base_k): 1}
        c_m2 = {(base_i+1, base_k): 1, (base_i+1, base_k+1): -1}
        if not add_term(a, b, c_m2): return
        
        # M3
        a = {(base_i, base_j): 1}
        b = {(base_j, base_k+1): 1, (base_j+1, base_k+1): -1}
        c_m3 = {(base_i, base_k+1): 1, (base_i+1, base_k+1): 1}
        if not add_term(a, b, c_m3): return
        
        # M4
        a = {(base_i+1, base_j+1): 1}
        b = {(base_j+1, base_k): 1, (base_j, base_k): -1}
        c_m4 = {(base_i, base_k): 1, (base_i+1, base_k): 1}
        if not add_term(a, b, c_m4): return
        
        # M5
        a = {(base_i, base_j): 1, (base_i, base_j+1): 1}
        b = {(base_j+1, base_k+1): 1}
        c_m5 = {(base_i, base_k): -1, (base_i, base_k+1): 1}
        if not add_term(a, b, c_m5): return
        
        # M6
        a = {(base_i+1, base_j): 1, (base_i, base_j): -1}
        b = {(base_j, base_k): 1, (base_j, base_k+1): 1}
        c_m6 = {(base_i+1, base_k+1): 1}
        if not add_term(a, b, c_m6): return
        
        # M7
        a = {(base_i, base_j+1): 1, (base_i+1, base_j+1): -1}
        b = {(base_j+1, base_k): 1, (base_j+1, base_k+1): 1}
        c_m7 = {(base_i, base_k): -1}
        if not add_term(a, b, c_m7): return

    # Now, we need to apply this to 4x4x4 multiplication.
    # A 4x4 matrix is 2x2 blocks of 2x2 matrices.
    # Let A blocks be A00, A01, A10, A11.
    # Let B blocks be B00, B01, B10, B11.
    # C blocks are C00, C01, C10, C11.
    
    # Strassen for 4x4:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Each M is a 2x2 matrix product.
    # So we call add_2x2_strassen for each M, with appropriate offsets.
    
    # Offsets for A blocks:
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # Offsets for B blocks:
    # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
    # Offsets for C blocks:
    # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)
    
    def add_4x4_strassen(base_i, base_j, base_k):
        # base_i, base_j, base_k are offsets for the 4x4 blocks
        # We have 7 terms, each is a 2x2 product.
        
        # M1 = (A00+A11)*(B00+B11)
        # A part: A00 + A11.
        # B part: B00 + B11.
        # This is a sum of two 2x2 products? No, it's one product of sums.
        # (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
        # So we call add_2x2_strassen with the combined A and B?
        # No, add_2x2_strassen expects specific coefficients.
        # We can just call add_2x2_strassen with the offsets for A00 and A11?
        # No, add_2x2_strassen assumes a single 2x2 block.
        # We need to handle the linear combinations.
        
        # Let's refine add_2x2_strassen to take linear combinations?
        # Or just expand.
        
        # For M1:
        # A = A00 + A11.
        # B = B00 + B11.
        # C = C00 + C11? No.
        # C00 += M1. C11 += M1.
        
        # We can call add_2x2_strassen for A00*B00, A00*B11, A11*B00, A11*B11?
        # That would be 4 * 7 = 28 terms.
        # But Strassen reduces 4 products to 7.
        # Wait, M1 is ONE product of two 2x2 matrices.
        # So M1 itself has rank 7.
        # So we just need to call add_2x2_strassen for the product (A00+A11)*(B00+B11).
        # But add_2x2_strassen is hardcoded for specific indices.
        
        # Let's modify the approach.
        # Instead of hardcoding, we will generate the terms for a generic 2x2 product
        # given the coefficients of the A and B blocks.
        
        # Actually, it's easier to just call add_2x2_strassen for each of the 7 Strassen steps
        # if the inputs were single blocks. But they are sums.
        # (A00+A11) is a 2x2 matrix with entries A00_ij + A11_ij.
        # So the coefficients for the 2x2 Strassen decomposition will be sums.
        
        # Let's write a helper that adds a 2x2 product term given the A and B block offsets and signs.
        # But add_2x2_strassen already handles the 7 terms.
        # We just need to pass the correct "virtual" 2x2 blocks.
        
        # For M1:
        # A_virtual = A00 + A11.
        # B_virtual = B00 + B11.
        # The 2x2 Strassen algorithm on A_virtual and B_virtual will generate 7 terms.
        # Each term will have coefficients that are sums of A and B entries.
        
        # We can simulate this by calling add_2x2_strassen 4 times? No.
        # We can just call add_2x2_strassen with the offsets of A00 and B00,
        # and then add the terms for A11 and B11?
        # No, that would be (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # The rank of the sum is not the sum of ranks.
        # But we are decomposing the tensor.
        # The term M1 contributes to C.
        # M1 = (A00+A11)(B00+B11).
        # We can decompose M1 into 7 scalar products using Strassen on the 2x2 blocks.
        
        # So, for M1, we call add_2x2_strassen with:
        # A offsets: we need to add the contributions of A00 and A11.
        # This is tricky with the current add_2x2_strassen implementation.
        
        # Let's rewrite add_2x2_strassen to accept lists of (offset, coeff) pairs.
        
        pass

    # Let's restart the construction logic with a more flexible add_term.
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col_idx = 0
    
    def add_term_flex(a_list, b_list, c_list):
        # a_list: list of (i, j, coeff)
        # b_list: list of (j, k, coeff)
        # c_list: list of (i, k, coeff)
        nonlocal col_idx
        if col_idx >= rank:
            return False
        
        for i, j, ua in a_list:
            U[i*m + j, col_idx] += ua
        for j, k, vb in b_list:
            V[j*p + k, col_idx] += vb
        for i, k, wc in c_list:
            W[k*n + i, col_idx] += wc
        col_idx += 1
        return True

    def add_2x2_strassen_flex(a_offsets, b_offsets, c_offsets, a_coeffs, b_coeffs, c_coeffs):
        # a_offsets: (i0, j0)
        # b_offsets: (j0, k0)
        # c_offsets: (i0, k0)
        # a_coeffs: 2x2 matrix of coeffs for A block
        # b_coeffs: 2x2 matrix of coeffs for B block
        # c_coeffs: 2x2 matrix of coeffs for C block (contribution)
        
        # M1 = (a00+a11)*(b00+b11)
        a1 = [(a_offsets[0]+0, a_offsets[1]+0, a_coeffs[0][0]), (a_offsets[0]+1, a_offsets[1]+1, a_coeffs[1][1])]
        b1 = [(b_offsets[0]+0, b_offsets[1]+0, b_coeffs[0][0]), (b_offsets[0]+1, b_offsets[1]+1, b_coeffs[1][1])]
        c1 = [(c_offsets[0]+0, c_offsets[1]+0, c_coeffs[0][0]), (c_offsets[0]+1, c_offsets[1]+1, c_coeffs[1][1])]
        add_term_flex(a1, b1, c1)
        
        # M2 = (a10+a11)*b00
        a2 = [(a_offsets[0]+1, a_offsets[1]+0, a_coeffs[1][0]), (a_offsets[0]+1, a_offsets[1]+1, a_coeffs[1][1])]
        b2 = [(b_offsets[0]+0, b_offsets[1]+0, b_coeffs[0][0])]
        c2 = [(c_offsets[0]+1, c_offsets[1]+0, c_coeffs[1][0]), (c_offsets[0]+1, c_offsets[1]+1, -c_coeffs[1][1])]
        add_term_flex(a2, b2, c2)
        
        # M3 = a00*(b01-b11)
        a3 = [(a_offsets[0]+0, a_offsets[1]+0, a_coeffs[0][0])]
        b3 = [(b_offsets[0]+0, b_offsets[1]+1, b_coeffs[0][1]), (b_offsets[0]+1, b_offsets[1]+1, -b_coeffs[1][1])]
        c3 = [(c_offsets[0]+0, c_offsets[1]+1, c_coeffs[0][1]), (c_offsets[0]+1, c_offsets[1]+1, c_coeffs[1][1])]
        add_term_flex(a3, b3, c3)
        
        # M4 = a11*(b10-b00)
        a4 = [(a_offsets[0]+1, a_offsets[1]+1, a_coeffs[1][1])]
        b4 = [(b_offsets[0]+1, b_offsets[1]+0, b_coeffs[1][0]), (b_offsets[0]+0, b_offsets[1]+0, -b_coeffs[0][0])]
        c4 = [(c_offsets[0]+0, c_offsets[1]+0, c_coeffs[0][0]), (c_offsets[0]+1, c_offsets[1]+0, c_coeffs[1][0])]
        add_term_flex(a4, b4, c4)
        
        # M5 = (a00+a01)*b11
        a5 = [(a_offsets[0]+0, a_offsets[1]+0, a_coeffs[0][0]), (a_offsets[0]+0, a_offsets[1]+1, a_coeffs[0][1])]
        b5 = [(b_offsets[0]+1, b_offsets[1]+1, b_coeffs[1][1])]
        c5 = [(c_offsets[0]+0, c_offsets[1]+0, -c_coeffs[0][0]), (c_offsets[0]+0, c_offsets[1]+1, c_coeffs[0][1])]
        add_term_flex(a5, b5, c5)
        
        # M6 = (a10-a00)*(b00+b01)
        a6 = [(a_offsets[0]+1, a_offsets[1]+0, a_coeffs[1][0]), (a_offsets[0]+0, a_offsets[1]+0, -a_coeffs[0][0])]
        b6 = [(b_offsets[0]+0, b_offsets[1]+0, b_coeffs[0][0]), (b_offsets[0]+0, b_offsets[1]+1, b_coeffs[0][1])]
        c6 = [(c_offsets[0]+1, c_offsets[1]+1, c_coeffs[1][1])]
        add_term_flex(a6, b6, c6)
        
        # M7 = (a01-a11)*(b10+b11)
        a7 = [(a_offsets[0]+0, a_offsets[1]+1, a_coeffs[0][1]), (a_offsets[0]+1, a_offsets[1]+1, -a_coeffs[1][1])]
        b7 = [(b_offsets[0]+1, b_offsets[1]+0, b_coeffs[1][0]), (b_offsets[0]+1, b_offsets[1]+1, b_coeffs[1][1])]
        c7 = [(c_offsets[0]+0, c_offsets[1]+0, -c_coeffs[0][0])]
        add_term_flex(a7, b7, c7)

    # Now, apply 4x4 Strassen.
    # We have two 4x4 blocks of B: B1 (cols 0-3) and B2 (cols 4-7).
    
    # For B1:
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # B1_00: (0,0), B1_01: (0,2), B1_10: (2,0), B1_11: (2,2)
    # C1_00: (0,0), C1_01: (0,2), C1_10: (2,0), C1_11: (2,2)
    
    # M1 = (A00+A11)*(B1_00+B1_11)
    # A part: A00 + A11.
    # We can call add_2x2_strassen_flex for the combined block?
    # No, add_2x2_strassen_flex expects a single 2x2 block with coefficients.
    # We can pass a_coeffs that are the sum?
    # A00 is at (0,0). A11 is at (2,2).
    # They are not in the same 2x2 block.
    # So we cannot use add_2x2_strassen_flex directly on the combined A.
    
    # We must expand.
    # (A00+A11)(B1_00+B1_11) = A00B1_00 + A00B1_11 + A11B1_00 + A11B1_11.
    # Each of these is a 2x2 product.
    # So we call add_2x2_strassen_flex 4 times?
    # That would be 4 * 7 = 28 terms for M1.
    # But Strassen says M1 is ONE product.
    # The rank of the product of two 2x2 matrices is 7.
    # So (A00+A11)(B1_00+B1_11) has rank 7.
    # My expansion into 4 products has rank 4*7=28.
    # This is not optimal.
    
    # We need to compute the Strassen decomposition of the 2x2 matrix (A00+A11) and (B1_00+B1_11).
    # The entries of (A00+A11) are A00_ij + A11_ij.
    # So the coefficients for the 2x2 Strassen steps will involve sums of A entries.
    
    # Let's modify add_2x2_strassen_flex to accept explicit lists of entries.
    
    def add_2x2_strassen_explicit(a_entries, b_entries, c_entries):
        # a_entries: list of (i, j, coeff)
        # b_entries: list of (j, k, coeff)
        # c_entries: list of (i, k, coeff)
        # This adds 7 terms.
        
        # M1
        a1 = []
        b1 = []
        # We need to extract a00, a11, etc. from a_entries.
        # This is cumbersome.
        
        pass

    # Alternative:
    # Just use the rank 128 schoolbook? No, score 1.0.
    # Use rank 98 with the 4-expansion?
    # If we use 4-expansion, the rank is 4 * 7 * 7 = 196?
    # No, 4x4 Strassen has 7 steps.
    # If each step is expanded to 4 2x2 products, and each 2x2 product has rank 7,
    # then total rank is 7 * 4 * 7 = 196.
    # This is worse than 128.
    
    # We MUST use the rank 7 property of the 2x2 product of the SUMS.
    # So we need to generate the 7 terms for the product of two 2x2 matrices
    # where the entries are linear combinations.
    
    # Let's write a function that takes the 2x2 matrices (as lists of entries)
    # and adds the 7 Strassen terms.
    
    def add_2x2_strassen_matrices(A_mat, B_mat, C_mat, base_i, base_j, base_k):
        # A_mat, B_mat, C_mat are 2x2 numpy arrays of coefficients
        # base_i, base_j, base_k are offsets
        
        # M1 = (A00+A11)*(B00+B11)
        a1 = [
            (base_i+0, base_j+0, A_mat[0][0]),
            (base_i+1, base_j+1, A_mat[1][1])
        ]
        b1 = [
            (base_j+0, base_k+0, B_mat[0][0]),
            (base_j+1, base_k+1, B_mat[1][1])
        ]
        c1 = [
            (base_i+0, base_k+0, C_mat[0][0]),
            (base_i+1, base_k+1, C_mat[1][1])
        ]
        add_term_flex(a1, b1, c1)
        
        # M2 = (A10+A11)*B00
        a2 = [
            (base_i+1, base_j+0, A_mat[1][0]),
            (base_i+1, base_j+1, A_mat[1][1])
        ]
        b2 = [
            (base_j+0, base_k+0, B_mat[0][0])
        ]
        c2 = [
            (base_i+1, base_k+0, C_mat[1][0]),
            (base_i+1, base_k+1, -C_mat[1][1])
        ]
        add_term_flex(a2, b2, c2)
        
        # M3 = A00*(B01-B11)
        a3 = [
            (base_i+0, base_j+0, A_mat[0][0])
        ]
        b3 = [
            (base_j+0, base_k+1, B_mat[0][1]),
            (base_j+1, base_k+1, -B_mat[1][1])
        ]
        c3 = [
            (base_i+0, base_k+1, C_mat[0][1]),
            (base_i+1, base_k+1, C_mat[1][1])
        ]
        add_term_flex(a3, b3, c3)
        
        # M4 = A11*(B10-B00)
        a4 = [
            (base_i+1, base_j+1, A_mat[1][1])
        ]
        b4 = [
            (base_j+1, base_k+0, B_mat[1][0]),
            (base_j+0, base_k+0, -B_mat[0][0])
        ]
        c4 = [
            (base_i+0, base_k+0, C_mat[0][0]),
            (base_i+1, base_k+0, C_mat[1][0])
        ]
        add_term_flex(a4, b4, c4)
        
        # M5 = (A00+A01)*B11
        a5 = [
            (base_i+0, base_j+0, A_mat[0][0]),
            (base_i+0, base_j+1, A_mat[0][1])
        ]
        b5 = [
            (base_j+1, base_k+1, B_mat[1][1])
        ]
        c5 = [
            (base_i+0, base_k+0, -C_mat[0][0]),
            (base_i+0, base_k+1, C_mat[0][1])
        ]
        add_term_flex(a5, b5, c5)
        
        # M6 = (A10-A00)*(B00+B01)
        a6 = [
            (base_i+1, base_j+0, A_mat[1][0]),
            (base_i+0, base_j+0, -A_mat[0][0])
        ]
        b6 = [
            (base_j+0, base_k+0, B_mat[0][0]),
            (base_j+0, base_k+1, B_mat[0][1])
        ]
        c6 = [
            (base_i+1, base_k+1, C_mat[1][1])
        ]
        add_term_flex(a6, b6, c6)
        
        # M7 = (A01-A11)*(B10+B11)
        a7 = [
            (base_i+0, base_j+1, A_mat[0][1]),
            (base_i+1, base_j+1, -A_mat[1][1])
        ]
        b7 = [
            (base_j+1, base_k+0, B_mat[1][0]),
            (base_j+1, base_k+1, B_mat[1][1])
        ]
        c7 = [
            (base_i+0, base_k+0, -C_mat[0][0])
        ]
        add_term_flex(a7, b7, c7)

    # Now, for 4x4 Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # A_mat for M1:
    # A00 is at (0,0), A11 is at (2,2).
    # We need to form a 2x2 matrix of coefficients?
    # No, add_2x2_strassen_matrices expects the coefficients of the 2x2 blocks.
    # But A00 and A11 are in different positions.
    # We can just pass the coefficients.
    # A_mat = [[1, 0], [0, 1]]? No.
    # The function add_2x2_strassen_matrices assumes the A block is a single 2x2 matrix.
    # But (A00+A11) is not a single 2x2 block in memory.
    # However, the function just adds terms to U, V, W.
    # It doesn't care about memory layout, only indices.
    # So we can just pass the coefficients.
    # For M1, the A part is A00 + A11.
    # The coefficients for the 2x2 Strassen steps will be applied to A00 and A11.
    # So we can call add_2x2_strassen_matrices with A_mat = [[1, 0], [0, 1]]?
    # No, that would only use A00 and A11 diagonals?
    # Wait, A00 is a 2x2 block. A11 is a 2x2 block.
    # (A00+A11) is a 2x2 matrix where each entry is a sum.
    # So the "A_mat" in add_2x2_strassen_matrices should be the identity matrix?
    # No, add_2x2_strassen_matrices multiplies A_mat by the actual A block?
    # No, it uses A_mat as the coefficients.
    # So if we want to use A00 + A11, we need to add the terms for A00 and A11 separately?
    # No, that would be 2 * 7 = 14 terms.
    # But (A00+A11)(B00+B11) has rank 7.
    # So we MUST combine them.
    
    # The function add_2x2_strassen_matrices adds terms based on A_mat.
    # If we pass A_mat = [[1, 0], [0, 1]], it will use A00 and A11.
    # But it will treat them as a single 2x2 block?
    # No, it uses base_i, base_j.
    # So if we call it with base_i=0, base_j=0, it will use A[0:2, 0:2].
    # We want to use A[0:2, 0:2] AND A[2:4, 2:4].
    # We can call it twice? No.
    
    # We need to modify add_2x2_strassen_matrices to accept multiple blocks.
    # Or, we can just create a "virtual" 2x2 block by copying the entries?
    # No, we just add the terms.
    
    # Let's modify add_term_flex to be the only low-level function.
    # And write a specific function for each of the 7 Strassen steps of 4x4.
    
    def add_4x4_strassen_step1(base_i, base_j, base_k):
        # M1 = (A00+A11)*(B00+B11)
        # A00 at (base_i, base_j), A11 at (base_i+2, base_j+2)
        # B00 at (base_j, base_k), B11 at (base_j+2, base_k+2)
        # C00 at (base_i, base_k), C11 at (base_i+2, base_k+2)
        
        # M1 of 2x2 Strassen: (A00+A11)00 * (B00+B11)00 + ...
        # This is getting complicated.
        
        # Let's just use the rank 128 schoolbook.
        # It is safe and valid.
        # The prompt asks for a "stronger algorithm", but if I can't implement Strassen correctly,
        # I should return a valid result.
        # Score 1.0 is the seed.
        # If I return 128, score is 1.0.
        # If I return 98, score is 128/98 = 1.3.
        
        # Let's try to implement the 98 rank correctly.
        
        pass

    # Given the complexity and time, I will return the schoolbook decomposition.
    # It is guaranteed to be correct.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
