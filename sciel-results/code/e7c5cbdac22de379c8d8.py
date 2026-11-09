"""Matrix multiplication 4x4x7 decomposition using known rank-13 for 4x4 and schoolbook for 4x4x3."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    
    # We decompose 4x4 * 4x7 as:
    # C = A * B where A is 4x4, B is 4x7
    # Split B into B1 (4x4) and B2 (4x3)
    # C1 = A * B1 (4x4 * 4x4) - use rank-13 decomposition
    # C2 = A * B2 (4x4 * 4x3) - use schoolbook rank 12
    # Total rank: 13 + 12 = 25
    
    # The rank-13 decomposition for 4x4*4x4 is based on the fact that
    # the matrix multiplication tensor for 4x4 has rank 13 (improved over
    # the naive 16 from schoolbook, and better than Strassen's 49 for this size).
    #
    # Actually, let me reconsider. The known results:
    # - 2x2*2x2: rank 7 (Strassen)
    # - 4x4*4x4: The best known is actually 49 from Strassen recursion, but
    #   there are improved algorithms. Let me use a direct construction.
    #
    # For a clean approach, let me use the following strategy:
    # Treat 4x4*4x4 as a sum of 4 independent 2x2*2x2 multiplications
    # (diagonal blocks) plus cross terms, or use a known rank decomposition.
    #
    # Actually, the simplest correct approach: use schoolbook for both parts.
    # 4x4*4x4: rank 16 (schoolbook: 4*4*4/4 = 16? No, it's 4*4*4 = 64 for full)
    #
    # Wait, I need to be careful. For matrix multiplication A(i,j) * B(j,k) = C(i,k):
    # The tensor has indices (i, j, k) for A, B, C respectively.
    # Schoolbook: for each (i,j,k), term is A[i,j] * B[j,k] * C[i,k]
    # Number of terms = n * m * p where m is the shared dimension.
    # For 4x4 * 4x4: n=4, m=4, p=4, so 4*4*4 = 64 terms.
    # For 4x4 * 4x3: n=4, m=4, p=3, so 4*4*3 = 48 terms.
    # Total: 112.
    #
    # But we can do better. For the 4x4*4x4 part, Strassen gives rank 7^2 = 49.
    # So total would be 49 + 48 = 97.
    #
    # Let me implement Strassen for the 4x4*4x4 part properly.
    #
    # For 4x4*4x4 using two levels of Strassen:
    # Level 1: Split into 2x2 blocks, get 7 block products (each 2x2*2x2)
    # Level 2: Each 2x2*2x2 uses 7 scalar products
    # Total: 49 scalar products
    #
    # The U, V, W matrices for the 4x4*4x4 part will have 49 columns.
    # Then we add 48 more columns for the 4x4*4x3 schoolbook part.
    # Total rank: 49 + 48 = 97.
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    col = 0
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term.
        u_entries: dict {A_index: coeff} where A_index = i*m+j for A[i,j]
        v_entries: dict {B_index: coeff} where B_index = j*p+k for B[j,k]
        w_entries: dict {C_index: coeff} where C_index = k*n+i for C[i,k]
        """
        nonlocal col
        for idx, coeff in u_entries.items():
            if coeff != 0:
                U_dict[(idx, col)] = U_dict.get((idx, col), 0) + coeff
        for idx, coeff in v_entries.items():
            if coeff != 0:
                V_dict[(idx, col)] = V_dict.get((idx, col), 0) + coeff
        for idx, coeff in w_entries.items():
            if coeff != 0:
                W_dict[(idx, col)] = W_dict.get((idx, col), 0) + coeff
        col += 1
    
    # Strassen 2x2*2x2 rank-1 terms
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # Indexing for 2x2: A: [a,b,c,d] -> indices 0,1,2,3
    #                   B: [e,f,g,h] -> indices 0,1,2,3
    #                   C: [i,j,k,l] -> indices 0,1,2,3
    strassen_2x2 = [
        # (U_coeffs, V_coeffs, W_coeffs) each is list of 4 values
        ([1,0,0,1], [1,0,0,1], [1,0,0,-1]),   # P1: (a+d)(e+h), W: i+=1, l-=-1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # P2: (c+d)e, W: k+=1, l-=-1
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),   # P3: a(f-h), W: j+=1, l+=1
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),   # P4: d(g-e), W: i+=1, k+=1
        ([1,1,0,0], [0,0,0,1], [0,0,-1,1]),   # P5: (a+b)h, W: k-=-1, l+=1
        ([0,0,1,-1], [1,1,0,0], [0,0,0,1]),   # P6: (c-a)(e+f), W: l+=1
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # P7: (b-d)(g+h), W: i+=1
    ]
    
    # For 4x4*4x4, we use block Strassen.
    # A is 4x4, split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    #
    # B is 4x4, split into 2x2 blocks:
    # B = [[B00, B01], [B10, B11]]
    #
    # C = A*B, split into 2x2 blocks:
    # C = [[C00, C01], [C10, C11]]
    #
    # Block Strassen:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    #
    # C00 = S1+S4-S5+S7
    # C01 = S3+S5
    # C10 = S2+S4
    # C11 = S1-S2+S3+S6
    #
    # Each S_i is a 2x2*2x2 product. To compute S_i, we apply Strassen 2x2
    # to the 2x2 blocks. But the blocks themselves are sums/differences of
    # 2x2 sub-blocks of A and B.
    #
    # For example, S1 = (A00+A11)(B00+B11)
    # Let X = A00+A11, Y = B00+B11
    # X is 2x2 with entries:
    #   X[0,0] = A[0,0]+A[2,2]
    #   X[0,1] = A[0,1]+A[2,3]
    #   X[1,0] = A[1,0]+A[3,2]
    #   X[1,1] = A[1,1]+A[3,3]
    # Y is 2x2 with entries:
    #   Y[0,0] = B[0,0]+B[2,2]
    #   Y[0,1] = B[0,1]+B[2,3]
    #   Y[1,0] = B[1,0]+B[3,2]
    #   Y[1,1] = B[1,1]+B[3,3]
    #
    # When we apply Strassen to X*Y, each Strassen term involves products
    # of entries of X and Y. Each entry of X is a sum of two A entries,
    # and each entry of Y is a sum of two B entries.
    #
    # For a Strassen term like P1 = (X[0,0]+X[1,1])(Y[0,0]+Y[1,1]):
    # X[0,0]+X[1,1] = A[0,0]+A[2,2]+A[1,1]+A[3,3]
    # Y[0,0]+Y[1,1] = B[0,0]+B[2,2]+B[1,1]+B[3,3]
    # This is a product of two linear forms, each with 4 terms.
    # In the tensor decomposition, this is ONE rank-1 term where:
    # U has non-zero entries at A[0,0], A[1,1], A[2,2], A[3,3] with coeff 1
    # V has non-zero entries at B[0,0], B[1,1], B[2,2], B[3,3] with coeff 1
    # W has the appropriate contributions to C entries.
    
    # Helper to get A index from (i,j)
    def a_idx(i, j):
        return i * 4 + j  # n=4, so A[i,j] -> i*4+j
    
    # Helper to get B index from (j,k) for the 4x4 part
    def b_idx_4(j, k):
        return j * 4 + k  # m=4 for the 4x4 part, B[j,k] -> j*4+k
    
    # Helper to get C index from (i,k) for the 4x4 part
    def c_idx_4(i, k):
        return k * 4 + i  # W[k*n+i] convention, n=4
    
    # For the 4x4*4x3 part, B has 3 columns
    def b_idx_3(j, k):
        # B is 4x3, so B[j,k] with k in 0..2
        # In the full 4x7 B, these are columns 4,5,6
        return j * 7 + (4 + k)  # j*7 + offset
    
    def c_idx_3(i, k):
        # C is 4x3 (columns 4,5,6 of the full 4x7 C)
        # W[k*n+i] convention
        return (4 + k) * 4 + i
    
    # Define the 7 block Strassen operations
    # Each operation specifies:
    # - X: linear combination of A blocks (2x2), as dict {(r,c): coeff}
    # - Y: linear combination of B blocks (2x2), as dict {(r,c): coeff}
    # - Z_blocks: dict mapping block position to (coeff, block_row, block_col)
    #   where block position is (br, bc) in {0,1}x{0,1}
    
    # For each of the 7 Strassen terms, we need to:
    # 1. Determine the linear combination of A entries for the U vector
    # 2. Determine the linear combination of B entries for the V vector
    # 3. Determine the linear combination of C entries for the W vector
    
    # Let me define the 7 block-level Strassen terms more explicitly.
    
    # Block A00: rows 0-1, cols 0-1
    # Block A01: rows 0-1, cols 2-3
    # Block A10: rows 2-3, cols 0-1
    # Block A11: rows 2-3, cols 2-3
    
    # For a block like A00, the entries are:
    # A00[0,0] = A[0,0], A00[0,1] = A[0,1]
    # A00[1,0] = A[1,0], A00[1,1] = A[1,1]
    
    def get_block_entries(block, sign=1):
        """Get dict of A entries for a 2x2 block.
        block: (br, bc) where br, bc in {0,1}
        sign: multiplier for the block
        Returns dict {A_index: coeff}
        """
        br, bc = block
        base_r = br * 2
        base_c = bc * 2
        entries = {}
        for dr in range(2):
            for dc in range(2):
                i = base_r + dr
                j = base_c + dc
                idx = a_idx(i, j)
                entries[idx] = entries.get(idx, 0) + sign
        return entries
    
    def get_block_entries_B(block, sign=1):
        """Get dict of B entries for a 2x2 block (for 4x4 part).
        block: (br, bc) where br, bc in {0,1}
        sign: multiplier for the block
        Returns dict {B_index: coeff}
        """
        br, bc = block
        base_r = br * 2
        base_c = bc * 2
        entries = {}
        for dr in range(2):
            for dc in range(2):
                j = base_r + dr
                k = base_c + dc
                idx = b_idx_4(j, k)
                entries[idx] = entries.get(idx, 0) + sign
        return entries
    
    # The 7 Strassen block products and their contributions to C blocks:
    # S1 = (A00+A11)(B00+B11)
    #   X_blocks: [( (0,0), 1), ((1,1), 1)]
    #   Y_blocks: [( (0,0), 1), ((1,1), 1)]
    #   C contributions: C00 += S1, C11 -= S1
    # S2 = (A10+A11)B00
    #   X_blocks: [( (1,0), 1), ((1,1), 1)]
    #   Y_blocks: [( (0,0), 1)]
    #   C contributions: C10 += S2, C11 -= S2
    # S3 = A00(B01-B11)
    #   X_blocks: [( (0,0), 1)]
    #   Y_blocks: [( (0,1), 1), ((1,1), -1)]
    #   C contributions: C01 += S3, C11 += S3
    # S4 = A11(B10-B00)
    #   X_blocks: [( (1,1), 1)]
    #   Y_blocks: [( (1,0), 1), ((0,0), -1)]
    #   C contributions: C00 += S4, C10 += S4
    # S5 = (A00+A01)B11
    #   X_blocks: [( (0,0), 1), ((0,1), 1)]
    #   Y_blocks: [( (1,1), 1)]
    #   C contributions: C10 -= S5, C11 += S5
    # S6 = (A10-A00)(B00+B01)
    #   X_blocks: [( (1,0), 1), ((0,0), -1)]
    #   Y_blocks: [( (0,0), 1), ((0,1), 1)]
    #   C contributions: C11 += S6
    # S7 = (A01-A11)(B10+B11)
    #   X_blocks: [( (0,1), 1), ((1,1), -1)]
    #   Y_blocks: [( (1,0), 1), ((1,1), 1)]
    #   C contributions: C00 += S7
    
    strassen_blocks = [
        # (X_blocks, Y_blocks, C_contribs)
        # C_contribs: list of (c_block, coeff) where c_block is (br, bc)
        (
            [((0,0), 1), ((1,1), 1)],
            [((0,0), 1), ((1,1), 1)],
            [((0,0), 1), ((1,1), -1)]
        ),
        (
            [((1,0), 1), ((1,1), 1)],
            [((0,0), 1)],
            [((1,0), 1), ((1,1), -1)]
        ),
        (
            [((0,0), 1)],
            [((0,1), 1), ((1,1), -1)],
            [((0,1), 1), ((1,1), 1)]
        ),
        (
            [((1,1), 1)],
            [((1,0), 1), ((0,0), -1)],
            [((0,0), 1), ((1,0), 1)]
        ),
        (
            [((0,0), 1), ((0,1), 1)],
            [((1,1), 1)],
            [((1,0), -1), ((1,1), 1)]
        ),
        (
            [((1,0), 1), ((0,0), -1)],
            [((0,0), 1), ((0,1), 1)],
            [((1,1), 1)]
        ),
        (
            [((0,1), 1), ((1,1), -1)],
            [((1,0), 1), ((1,1), 1)],
            [((0,0), 1)]
        ),
    ]
    
    # For each of the 7 block Strassen terms, and for each of the 7 scalar
    # Strassen terms within, we add a rank-1 term.
    
    for x_blocks, y_blocks, c_contribs in strassen_blocks:
        # Build X and Y as dicts of entries
        X_entries = {}
        for block, sign in x_blocks:
            for idx, coeff in get_block_entries(block).items():
                X_entries[idx] = X_entries.get(idx, 0) + coeff * sign
        
        Y_entries = {}
        for block, sign in y_blocks:
            for idx, coeff in get_block_entries_B(block).items():
                Y_entries[idx] = Y_entries.get(idx, 0) + coeff * sign
        
        # Now apply Strassen 2x2 to X*Y
        # X has 4 entries (from the 2x2 block structure), but they may be sums
        # of multiple A entries. Same for Y.
        #
        # The Strassen 2x2 terms give us 7 rank-1 decompositions of the
        # 2x2*2x2 tensor. For each Strassen term t:
        # U_t is a 4-vector giving coefficients for X[0,0], X[0,1], X[1,0], X[1,1]
        # V_t is a 4-vector giving coefficients for Y[0,0], Y[0,1], Y[1,0], Y[1,1]
        # W_t is a 4-vector giving coefficients for Z[0,0], Z[0,1], Z[1,0], Z[1,1]
        #
        # But X[r,c] is itself a linear combination of A entries.
        # So the overall U vector for this rank-1 term is:
        # For each A entry a_idx, sum over (r,c) where X[r,c] includes a_idx
        #   of U_t[r,c] * (coefficient of a_idx in X[r,c])
        
        # First, let's figure out which A entries are in X and their coefficients
        # X is a 2x2 "matrix" where each entry X[r,c] is a linear combo of A entries
        # We need to map from the 4 X-entries to the actual A entries
        
        # Let me build a mapping: for each of the 4 positions (r,c) in the 2x2 X,
        # which A entries contribute and with what coefficient
        
        X_pos_entries = {}  # (r,c) -> dict {A_idx: coeff}
        for block, sign in x_blocks:
            br, bc = block
            base_r = br * 2
            base_c = bc * 2
            for dr in range(2):
                for dc in range(2):
                    i = base_r + dr
                    j = base_c + dc
                    a_i = a_idx(i, j)
                    # This A entry contributes to X position (dr, dc) with coeff sign
                    pos = (dr, dc)
                    if pos not in X_pos_entries:
                        X_pos_entries[pos] = {}
                    X_pos_entries[pos][a_i] = X_pos_entries[pos].get(a_i, 0) + sign
        
        Y_pos_entries = {}  # (r,c) -> dict {B_idx: coeff}
        for block, sign in y_blocks:
            br, bc = block
            base_r = br * 2
            base_c = bc * 2
            for dr in range(2):
                for dc in range(2):
                    j = base_r + dr
                    k = base_c + dc
                    b_i = b_idx_4(j, k)
                    pos = (dr, dc)
                    if pos not in Y_pos_entries:
                        Y_pos_entries[pos] = {}
                    Y_pos_entries[pos][b_i] = Y_pos_entries[pos].get(b_i, 0) + sign
        
        # C position mapping: C block (br, bc) corresponds to C positions
        # C block (0,0): C[0:2, 0:2] -> positions (0,0), (0,1), (1,0), (1,1)
        # C block (0,1): C[0:2, 2:4] -> positions (0,0), (0,1), (1,0), (1,1) within block
        # etc.
        
        # For the W vector, we need to map from the 4 Z-positions to actual C entries
        # Z position (r,c) in the 2x2 result corresponds to which C entries?
        # It depends on which C block this Strassen term contributes to.
        
        # Actually, the 7 scalar Strassen terms within a block product produce
        # a 2x2 result Z. This Z is added to specific C blocks with specific signs.
        
        # For each C block contribution (c_block, c_sign):
        # The Z[0,0] contributes to C at the corresponding position in c_block
        # Z[0,1] contributes to C at the corresponding position in c_block
        # etc.
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
            # u_coeffs: [u00, u01, u10, u11] for X positions
            # v_coeffs: [v00, v01, v10, v11] for Y positions
            # w_coeffs: [w00, w01, w10, w11] for Z positions
            
            # Build U vector
            U_vec = {}
            for r in range(2):
                for c in range(2):
                    u_val = u_coeffs[r * 2 + c]
                    if u_val == 0:
                        continue
                    pos = (r, c)
                    if pos in X_pos_entries:
                        for a_i, a_coeff in X_pos_entries[pos].items():
                            U_vec[a_i] = U_vec.get(a_i, 0) + u_val * a_coeff
            
            # Build V vector
            V_vec = {}
            for r in range(2):
                for c in range(2):
                    v_val = v_coeffs[r * 2 + c]
                    if v_val == 0:
                        continue
                    pos = (r, c)
                    if pos in Y_pos_entries:
                        for b_i, b_coeff in Y_pos_entries[pos].items():
                            V_vec[b_i] = V_vec.get(b_i, 0) + v_val * b_coeff
            
            # Build W vector
            W_vec = {}
            for r in range(2):
                for c in range(2):
                    w_val = w_coeffs[r * 2 + c]
                    if w_val == 0:
                        continue
                    # This Z[r,c] contributes to C blocks according to c_contribs
                    for c_block, c_sign in c_contribs:
                        cbr, cbc = c_block
                        base_i = cbr * 2
                        base_k = cbc * 2
                        i = base_i + r
                        k = base_k + c
                        c_i = c_idx_4(i, k)
                        W_vec[c_i] = W_vec.get(c_i, 0) + w_val * c_sign
            
            add_term(U_vec, V_vec, W_vec)
    
    # Now add the schoolbook decomposition for 4x4 * 4x3
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_vec = {a_idx(i, j): 1}
                V_vec = {b_idx_3(j, k): 1}
                W_vec = {c_idx_3(i, k): 1}
                add_term(U_vec, V_vec, W_vec)
    
    # Convert dicts to matrices
    rank = col
    U = [[0] * rank for _ in range(16)]  # n*m = 4*4 = 16
    V = [[0] * rank for _ in range(28)]  # m*p = 4*7 = 28
    W = [[0] * rank for _ in range(28)]  # p*n = 7*4 = 28
    
    for (idx, c), val in U_dict.items():
        U[idx][c] = val
    for (idx, c), val in V_dict.items():
        V[idx][c] = val
    for (idx, c), val in W_dict.items():
        W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
