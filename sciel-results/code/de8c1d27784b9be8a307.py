import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # We will build U, V, W as lists of lists or numpy arrays
    # Rank will be dynamic
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeffs=1.0):
        # u_indices: list of (idx, coeff)
        # v_indices: list of (idx, coeff)
        # w_indices: list of (idx, coeff)
        # This adds ONE rank-1 term to the decomposition
        pass

    # Actually, it's easier to just append rows to U, V, W matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will store them as lists of lists, then convert or keep as lists.
    # The return type is {'U': U, 'V': V, 'W': W} where U, V, W are lists of lists (or arrays).
    # The seed returns lists of lists. I will return lists of lists.
    
    U = []
    V = []
    W = []
    
    def add_rank1_term(u_vec, v_vec, w_vec):
        """
        u_vec: list of length n*m
        v_vec: list of length m*p
        w_vec: list of length p*n
        """
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # --- Part 1: 4x4x4 Multiplication (Columns 0-3 of B) ---
    # Using Recursive Strassen for 4x4x4
    
    # Base case: 2x2x2 Strassen
    def strassen_2x2_2x2(A_offsets, B_offsets, C_offsets):
        """
        Adds 7 rank-1 terms for a 2x2x2 multiplication.
        A_offsets: list of 4 indices for A (row-major 2x2)
        B_offsets: list of 4 indices for B (row-major 2x2)
        C_offsets: list of 4 indices for C (row-major 2x2)
        """
        a00, a01, a10, a11 = A_offsets
        b00, b01, b10, b11 = B_offsets
        c00, c01, c10, c11 = C_offsets
        
        terms = [
            # M1 = A00 * (B00 + B10) -> C00 += M1, C10 -= M1
            [(a00, 1), (b00, 1), (c00, 1)],
            [(a00, 1), (b10, 1), (c10, -1)], # Wait, M1 = A00*(B00+B10).
            # Let's use standard Strassen terms:
            # M1 = A00 * (B00 + B11) -> C00 += M1, C11 -= M1
            # M2 = (A10 + A11) * B00 -> C10 += M2, C11 += M2
            # M3 = (A00 + A01) * B11 -> C00 += M3, C01 -= M3
            # M4 = A10 * (B01 - B11) -> C10 += M4
            # M5 = A01 * (B10 + B11) -> C01 += M5
            # M6 = (A00 - A10) * (B10 + B11) -> C10 -= M6, C11 += M6
            # M7 = (A11 - A01) * (B00 + B01) -> C00 += M7, C11 -= M7
            
            # Let's implement exactly these.
        ]
        
        # M1
        add_rank1_term(
            _make_vec(n*m, [(a00, 1)]),
            _make_vec(m*p, [(b00, 1), (b11, 1)]),
            _make_vec(p*n, [(c00, 1), (c11, -1)])
        )
        # M2
        add_rank1_term(
            _make_vec(n*m, [(a10, 1), (a11, 1)]),
            _make_vec(m*p, [(b00, 1)]),
            _make_vec(p*n, [(c10, 1), (c11, 1)])
        )
        # M3
        add_rank1_term(
            _make_vec(n*m, [(a00, 1), (a01, 1)]),
            _make_vec(m*p, [(b11, 1)]),
            _make_vec(p*n, [(c00, 1), (c01, -1)])
        )
        # M4
        add_rank1_term(
            _make_vec(n*m, [(a10, 1)]),
            _make_vec(m*p, [(b01, 1), (b11, -1)]),
            _make_vec(p*n, [(c10, 1)])
        )
        # M5
        add_rank1_term(
            _make_vec(n*m, [(a01, 1)]),
            _make_vec(m*p, [(b10, 1), (b11, 1)]),
            _make_vec(p*n, [(c01, 1)])
        )
        # M6
        add_rank1_term(
            _make_vec(n*m, [(a00, 1), (a10, -1)]),
            _make_vec(m*p, [(b10, 1), (b11, 1)]),
            _make_vec(p*n, [(c10, -1), (c11, 1)])
        )
        # M7
        add_rank1_term(
            _make_vec(n*m, [(a11, 1), (a01, -1)]),
            _make_vec(m*p, [(b00, 1), (b01, 1)]),
            _make_vec(p*n, [(c00, 1), (c11, -1)])
        )

    def _make_vec(length, entries):
        vec = [0] * length
        for idx, coeff in entries:
            vec[idx] = coeff
        return vec

    # Recursive function for 4x4x4
    # It will call strassen_2x2_2x2 7 times, but with "scalar" entries that are themselves vectors?
    # No, we are doing scalar decomposition.
    # The 4x4x4 Strassen decomposes into 7 block multiplications of 2x2x2.
    # Each 2x2x2 block multiplication is itself decomposed into 7 scalar multiplications.
    # So we need to map the "block" indices to "scalar" indices.
    
    # Let's define the offsets for the 4x4 matrices.
    # A (4x4):
    # A00: A[0:2, 0:2] -> indices 0,1,4,5
    # A01: A[0:2, 2:4] -> indices 2,3,6,7
    # A10: A[2:4, 0:2] -> indices 8,9,12,13
    # A11: A[2:4, 2:4] -> indices 10,11,14,15
    
    # B (4x4 part, cols 0-3):
    # B00: B[0:2, 0:2] -> indices 0,1,8,9
    # B01: B[0:2, 2:4] -> indices 2,3,10,11
    # B10: B[2:4, 0:2] -> indices 16,17,24,25
    # B11: B[2:4, 2:4] -> indices 18,19,26,27
    
    # C (4x4 part, cols 0-3):
    # C00: C[0:2, 0:2] -> W indices?
    # W index is k*n + i.
    # C[i,k] -> W[k*4 + i].
    # C00 (i=0,1; k=0,1):
    # (0,0)->0, (1,0)->4, (0,1)->1, (1,1)->5.
    # Indices: 0, 1, 4, 5.
    # C01 (i=0,1; k=2,3):
    # (0,2)->2, (1,2)->6, (0,3)->3, (1,3)->7.
    # Indices: 2, 3, 6, 7.
    # C10 (i=2,3; k=0,1):
    # (2,0)->8, (3,0)->12, (2,1)->9, (3,1)->13.
    # Indices: 8, 9, 12, 13.
    # C11 (i=2,3; k=2,3):
    # (2,2)->10, (3,2)->14, (2,3)->11, (3,3)->15.
    # Indices: 10, 11, 14, 15.

    # The 7 Strassen block terms for 4x4x4:
    # S1 = A00 * (B00 + B11) -> C00 += S1, C11 -= S1
    # S2 = (A10 + A11) * B00 -> C10 += S2, C11 += S2
    # S3 = (A00 + A01) * B11 -> C00 += S3, C01 -= S3
    # S4 = A10 * (B01 - B11) -> C10 += S4
    # S5 = A01 * (B10 + B11) -> C01 += S5
    # S6 = (A00 - A10) * (B10 + B11) -> C10 -= S6, C11 += S6
    # S7 = (A11 - A01) * (B00 + B01) -> C00 += S7, C11 -= S7

    # For each S_k, we have a product of two 2x2 matrices (linear combos).
    # We must decompose this 2x2x2 multiplication into 7 scalar terms.
    # So for S1, we define:
    # A_part = A00
    # B_part = B00 + B11
    # C_part = C00 - C11 (with signs)
    
    # We need a function to add the 7 terms for a generic 2x2x2 multiplication
    # where the "entries" are lists of (index, coeff).
    
    def add_strassen_block(A_entries, B_entries, C_entries):
        """
        A_entries: list of 4 lists of (idx, coeff)
        B_entries: list of 4 lists of (idx, coeff)
        C_entries: list of 4 lists of (idx, coeff)
        """
        # M1
        u_vec = _make_vec(n*m, _flatten(A_entries[0]))
        v_vec = _make_vec(m*p, _flatten(B_entries[0]) + _flatten(B_entries[3]))
        w_vec = _make_vec(p*n, _flatten(C_entries[0]) + _flatten(C_entries[3])) # C00 + C11? No, C00 += M1, C11 -= M1
        # Wait, my C_entries should already have the signs for the target C.
        # Let's pass the combined C entries with correct signs.
        pass

    # This is getting complicated to manage symbolically.
    # Given the time, I will provide the rank 65 solution by concatenating the two parts.
    # The 4x4x4 part is implemented via the 7 recursive calls to 2x2x2.
    
    # To keep the code simple and correct, I will explicitly generate the 49 terms for 4x4x4
    # and 16 terms for 4x4x1.
    
    # --- Generate 4x4x4 terms (Rank 49) ---
    # Indices for A (16), B (16 for first 4 cols), C (16 for first 4 cols)
    
    # Define the 7 block multiplications
    # Each block mult is 2x2x2.
    # We iterate through the 7 block terms, and for each, we generate 7 scalar terms.
    
    # Block Term 1: A00 * (B00+B11) -> C00 +, C11 -
    # A00 indices: 0,1,4,5
    # B00+B11 indices: B00(0,1,8,9) + B11(18,19,26,27)
    # C00 - C11 indices: C00(0,1,4,5) - C11(10,11,14,15)
    
    # I will write a helper to expand one 2x2x2 block multiplication given the "block" indices.
    
    def expand_2x2x2(a_idxs, b_idxs, c_idxs):
        """
        a_idxs: list of 4 indices in U (0-15)
        b_idxs: list of 4 indices in V (0-19) -- Note: V is 4x5, so 20 entries.
                 For the first 4 cols, we use 0-15.
        c_idxs: list of 4 indices in W (0-19) -- For first 4 cols, we use 0-15.
        """
        # M1: A00 * (B00+B11)
        add_rank1_term(
            _make_vec(16, [(a_idxs[0], 1)]),
            _make_vec(20, [(b_idxs[0], 1), (b_idxs[3], 1)]),
            _make_vec(20, [(c_idxs[0], 1), (c_idxs[3], -1)])
        )
        # M2: (A10+A11) * B00
        add_rank1_term(
            _make_vec(16, [(a_idxs[2], 1), (a_idxs[3], 1)]),
            _make_vec(20, [(b_idxs[0], 1)]),
            _make_vec(20, [(c_idxs[2], 1), (c_idxs[3], 1)])
        )
        # M3: (A00+A01) * B11
        add_rank1_term(
            _make_vec(16, [(a_idxs[0], 1), (a_idxs[1], 1)]),
            _make_vec(20, [(b_idxs[3], 1)]),
            _make_vec(20, [(c_idxs[0], 1), (c_idxs[1], -1)])
        )
        # M4: A10 * (B01-B11)
        add_rank1_term(
            _make_vec(16, [(a_idxs[2], 1)]),
            _make_vec(20, [(b_idxs[1], 1), (b_idxs[3], -1)]),
            _make_vec(20, [(c_idxs[2], 1)])
        )
        # M5: A01 * (B10+B11)
        add_rank1_term(
            _make_vec(16, [(a_idxs[1], 1)]),
            _make_vec(20, [(b_idxs[2], 1), (b_idxs[3], 1)]),
            _make_vec(20, [(c_idxs[1], 1)])
        )
        # M6: (A00-A10) * (B10+B11)
        add_rank1_term(
            _make_vec(16, [(a_idxs[0], 1), (a_idxs[2], -1)]),
            _make_vec(20, [(b_idxs[2], 1), (b_idxs[3], 1)]),
            _make_vec(20, [(c_idxs[2], -1), (c_idxs[3], 1)])
        )
        # M7: (A11-A01) * (B00+B01)
        add_rank1_term(
            _make_vec(16, [(a_idxs[3], 1), (a_idxs[1], -1)]),
            _make_vec(20, [(b_idxs[0], 1), (b_idxs[1], 1)]),
            _make_vec(20, [(c_idxs[0], 1), (c_idxs[3], -1)])
        )

    # Define the 7 block calls for 4x4x4
    # A blocks:
    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]
    
    # B blocks (cols 0-3):
    B00 = [0, 1, 8, 9]
    B01 = [2, 3, 10, 11]
    B10 = [16, 17, 24, 25]
    B11 = [18, 19, 26, 27]
    
    # C blocks (cols 0-3):
    C00 = [0, 1, 4, 5]
    C01 = [2, 3, 6, 7]
    C10 = [8, 9, 12, 13]
    C11 = [10, 11, 14, 15]
    
    # Block 1: A00 * (B00+B11) -> C00 +, C11 -
    # We need to merge B00 and B11 indices for the B part of the 2x2x2 call?
    # No, the 2x2x2 call expects 4 distinct "block" indices.
    # But B00+B11 is a sum.
    # So for the 2x2x2 expansion, the "B" matrix is:
    # B00' = B00 + B11 (indices from both)
    # B01' = 0
    # B10' = 0
    # B11' = 0
    # This doesn't work with the fixed 7-term structure which assumes non-zero entries.
    
    # Correct approach:
    # The Strassen algorithm operates on linear combinations.
    # For Block 1, the "effective" B matrix has:
    # B_eff[0,0] = B00 + B11
    # B_eff[0,1] = 0
    # B_eff[1,0] = 0
    # B_eff[1,1] = 0
    # This is not a full 2x2 matrix.
    
    # However, we can just define the 7 terms of the 4x4x4 Strassen directly at the scalar level?
    # No, that's 7 terms.
    # But each term is a product of two 2x2 matrices.
    # So we MUST expand each 2x2x2.
    
    # Let's define a helper to pass "linear combination" indices.
    # For Block 1:
    # A_part: A00
    # B_part: B00 + B11
    # C_part: C00 - C11
    
    # In the 2x2x2 expansion for Block 1:
    # M1: A00[0,0] * (B_eff[0,0] + B_eff[1,1])
    # Since B_eff is sparse, many terms will be 0.
    
    # To simplify, I will implement the 7 Strassen terms for 4x4x4 directly.
    # Term 1: A00 * (B00+B11)
    # This product is 2x2.
    # Let X = A00, Y = B00+B11.
    # Compute XY using 7 Strassen multiplications on the entries of X and Y.
    # But Y has non-zero entries at all 4 positions?
    # B00 has non-zeros at 0,0. B11 has non-zeros at 1,1.
    # So Y[0,0] = B00, Y[1,1] = B11. Y[0,1]=0, Y[1,0]=0.
    # So Y is diagonal.
    # Multiplication of 2x2 by diagonal 2x2 can be done in 4 multiplications?
    # X * Y = [[X00*Y00, 0], [X10*Y11, 0]]? No.
    # Y = [[Y00, 0], [0, Y11]].
    # X*Y = [[X00*Y00, X01*Y11], [X10*Y00, X11*Y11]].
    # This requires 4 multiplications.
    # So Block 1 contributes 4 terms, not 7.
    
    # This optimization reduces the rank!
    # Block 1: A00 * (B00+B11). Y is diagonal. Rank 4.
    # Block 2: (A10+A11) * B00. B00 is non-zero only at [0,0].
    # Z = A10+A11. W = B00.
    # Z*W = [[Z00*B00, 0], [Z10*B00, 0]].
    # Rank 2? No, Z is 2x2, W is 2x2.
    # W = [[W00, 0], [0, 0]].
    # Z*W = [[Z00*W00, 0], [Z10*W00, 0]].
    # 2 multiplications.
    
    # This analysis shows we can do better than 49.
    # However, implementing this sparsity-aware Strassen is complex.
    # Given the constraint, I will stick to the standard 49 rank for 4x4x4.
    # It is safe and correct.
    
    # I will use the generic 7-term expansion for each block, even if some terms are zero.
    # This results in rank 49.
    
    # To make it work, I need to pass the "combined" indices to the 2x2x2 expander.
    # For Block 1:
    # A: A00
    # B: B00+B11 (all 4 positions? No, B00 is top-left, B11 is bottom-right)
    # So B_eff[0,0] = B00, B_eff[1,1] = B11.
    # In the 2x2x2 expander, we pass the indices for these positions.
    
    # I will refactor expand_2x2x2 to accept lists of indices for each of the 4 positions.
    
    # Block 1:
    # A: [A00, A01, A10, A11] -> [A00, 0, 0, 0] ? No, A is A00.
    # The block multiplication is A00 * B_eff.
    # A00 is a 2x2 matrix.
    # B_eff is a 2x2 matrix.
    # So we call expand_2x2x2(A00_idxs, B_eff_idxs, C_eff_idxs).
    
    A00_idxs = A00
    B_eff1 = [B00, [], [], B11] # B00 at 0,0; B11 at 1,1
    C_eff1 = [C00, [], [], C11] # C00 at 0,0; C11 at 1,1 (with sign?)
    # Signs are handled inside the 7-term expansion?
    # No, the 7-term expansion assumes C = A*B.
    # If C_eff has -1 for C11, we pass indices with -1 coeff?
    # My _make_vec handles coeffs.
    
    # I will create the terms manually for the 4x4x4 part to ensure correctness.
    # This will be long but reliable.
    
    return {"U": U, "V": V, "W": W}

# Note: The above thinking process was interrupted. I will now write the full code.
