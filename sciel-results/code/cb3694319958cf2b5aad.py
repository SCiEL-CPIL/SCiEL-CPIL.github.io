"""Block-Strassen decomposition for 2x4 x 4x4 matrix multiplication, rank 28."""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4  # A is 2x4, B is 4x4, C is 2x4
    
    # Split indices:
    # A: rows i in {0,1} (L/R), cols j in {0,1,2,3} (L={0,1}, R={2,3})
    # B: rows j in {0,1,2,3} (L={0,1}, R={2,3}), cols k in {0,1,2,3} (L={0,1}, R={2,3})
    # C: rows i in {0,1} (L/R), cols k in {0,1,2,3} (L={0,1}, R={2,3})
    # W index: W[k*n + i, r] -> W[k*2 + i, r]
    
    # 7 Strassen products per 2x2x2 block:
    # M1 = A_L * B_L
    # M2 = A_R * (B_L + B_R)
    # M3 = (A_L + A_R) * B_R
    # M4 = A_L * (B_R - B_L)
    # M5 = (A_L - A_R) * (B_L + B_R)
    # M6 = (A_R - A_L) * B_L
    # M7 = (A_L + A_R) * (B_R - B_L)
    
    # C_L = M1 + M2 + M5 - M7
    # C_R = M3 + M4 + M5 + M6
    
    # We have 4 blocks:
    # Block A: A_L * B_L -> contributes to C_L
    # Block B: A_R * B_L -> contributes to C_L
    # Block C: A_L * B_R -> contributes to C_R
    # Block D: A_R * B_R -> contributes to C_R
    
    # Each 2x2x2 Strassen decomposition has rank 7.
    # For 4 blocks, naive would be 4*7 = 28.
    # But we can share some terms!
    
    # Actually, let's think about this more carefully.
    # The 4 blocks are independent 2x2x2 multiplications.
    # Each has rank 7, so total rank = 4*7 = 28.
    
    # Let me construct the explicit decomposition.
    
    # Index mappings:
    # U row: i*m + j = i*4 + j, where i in {0,1}, j in {0,1,2,3}
    # V row: j*p + k = j*4 + k, where j in {0,1,2,3}, k in {0,1,2,3}
    # W row: k*n + i = k*2 + i, where k in {0,1,2,3}, i in {0,1}
    
    # For each 2x2x2 block, we have 7 Strassen terms.
    # Let's define the 4 blocks and their Strassen decompositions.
    
    # Block (a, b) where a in {L,R} for A's column split, b in {L,R} for B's column split
    # A_block_a: rows i in {0,1}, cols j in a_block
    # B_block_b: rows j in b_block, cols k in {0,1} for L or {2,3} for R... 
    
    # Wait, I need to be more careful. Let me reconsider.
    
    # Actually, the cleanest way: each 2x2x2 subproblem uses 7 multiplications.
    # We have 4 such subproblems (A_L*B_L, A_R*B_L, A_L*B_R, A_R*B_R), each producing
    # a 2x2 result. Total: 4*7 = 28 rank.
    
    # Let me just build it directly.
    
    # Define the Strassen formula for a generic 2x2x2:
    # Inputs: A (2x2), B (2x2), Output C (2x2)
    # M1 = A[0,:]*B[0,:]
    # M2 = A[1,:]*(B[0,:]+B[1,:])
    # M3 = (A[0,:]+A[1,:])*B[1,:]
    # M4 = A[0,:]*(B[1,:]-B[0,:])
    # M5 = (A[0,:]-A[1,:])*(B[0,:]+B[1,:])
    # M6 = (A[1,:]-A[0,:])*B[0,:]
    # M7 = (A[0,:]+A[1,:])*(B[1,:]-B[0,:])
    # C[0,:] = M1 + M2 + M5 - M7
    # C[1,:] = M3 + M4 + M5 + M6
    
    # For each of the 4 blocks, I'll assign 7 columns.
    
    # Block 0: A_L * B_L, where A_L = A[:, 0:2], B_L = B[0:2, 0:2], C contributes to C[:, 0:2]
    # Block 1: A_R * B_L, where A_R = A[:, 2:4], B_L = B[0:2, 0:2], C contributes to C[:, 0:2]
    # Block 2: A_L * B_R, where A_L = A[:, 0:2], B_R = B[0:2, 2:4], C contributes to C[:, 2:4]
    # Block 3: A_R * B_R, where A_R = A[:, 2:4], B_R = B[0:2, 2:4], C contributes to C[:, 2:4]
    
    # Wait, B is 4x4. B_L = B[0:2, :4]? No, that doesn't make sense for Strassen.
    
    # Let me reconsider. For the 2x2x2 Strassen, both A and B must be 2x2.
    # A is 2x4. If I split A's columns into L={0,1} and R={2,3}, each half is 2x2.
    # B is 4x4. If I split B's rows into L={0,1} and R={2,3}, and B's cols into L={0,1} and R={2,3},
    # then each 2x2 submatrix of B is 2x2.
    
    # So the 4 blocks are:
    # Block 0: A_L (2x2) * B[0:2, 0:2] (2x2) -> C_L (2x2)
    # Block 1: A_R (2x2) * B[0:2, 0:2] (2x2) -> C_L (2x2)
    # Block 2: A_L (2x2) * B[0:2, 2:4] (2x2) -> C_R (2x2)
    # Block 3: A_R (2x2) * B[0:2, 2:4] (2x2) -> C_R (2x2)
    
    # Hmm, but B's row split is only {0,1} vs {2,3}. That means we're only using B[0:2, :] and not B[2:4, :].
    # That's wrong. We need to use all of B.
    
    # Let me reconsider the splitting. 
    # A: 2x4. Split columns: L={0,1}, R={2,3}. Each 2x2.
    # B: 4x4. Split rows: L={0,1}, R={2,3}. Split cols: L={0,1}, R={2,3}. Each 2x2.
    # C: 2x4. Split cols: L={0,1}, R={2,3}. Each 2x2.
    
    # The 4 products are:
    # A_L * B_LL -> C_L
    # A_L * B_LR -> C_R
    # A_R * B_RL -> C_L
    # A_R * B_RR -> C_R
    
    # Where B_LL = B[0:2, 0:2], B_LR = B[0:2, 2:4], B_RL = B[2:4, 0:2], B_RR = B[2:4, 2:4].
    
    # Each is a 2x2x2 Strassen problem with rank 7. Total rank = 4*7 = 28.
    
    # Let me build this.
    
    rank = 28
    U = np.zeros((n * m, rank))  # (8, 28)
    V = np.zeros((m * p, rank))  # (16, 28)
    W = np.zeros((p * n, rank))  # (8, 28)
    
    # Define the 4 blocks
    # Block: (A_col_block, B_row_block, B_col_block)
    # A_col_block: 0=L, 1=R
    # B_row_block: 0=L, 1=R  
    # B_col_block: 0=L, 1=R
    
    # Strassen 7 terms for a 2x2x2:
    # term 0: M1 = A_top * B_top
    # term 1: M2 = A_bot * (B_top + B_bot)
    # term 2: M3 = (A_top + A_bot) * B_bot
    # term 3: M4 = A_top * (B_bot - B_top)
    # term 4: M5 = (A_top - A_bot) * (B_top + B_bot)
    # term 5: M6 = (A_bot - A_top) * B_top
    # term 6: M7 = (A_top + A_bot) * (B_bot - B_top)
    # C_top = M1 + M2 + M5 - M7
    # C_bot = M3 + M4 + M5 + M6
    
    # For each block, I need to map:
    # A indices: i in {0,1}, j in A_col_block's 2 columns
    # B indices: j in B_row_block's 2 rows, k in B_col_block's 2 columns
    # C indices: i in {0,1}, k in B_col_block's 2 columns (since C's col split matches B's col split)
    
    blocks = [
        (0, 0, 0),  # A_L, B_LL -> C_L
        (0, 0, 1),  # A_L, B_LR -> C_R
        (1, 1, 0),  # A_R, B_RL -> C_L
        (1, 1, 1),  # A_R, B_RR -> C_R
    ]
    
    col = 0
    for a_blk, b_row_blk, b_col_blk in blocks:
        # A's columns: if a_blk=0, cols 0,1; if a_blk=1, cols 2,3
        a_cols = [a_blk * 2, a_blk * 2 + 1]
        # B's rows: if b_row_blk=0, rows 0,1; if b_row_blk=1, rows 2,3
        b_rows = [b_row_blk * 2, b_row_blk * 2 + 1]
        # B's cols / C's cols: if b_col_blk=0, cols 0,1; if b_col_blk=1, cols 2,3
        b_cols = [b_col_blk * 2, b_col_blk * 2 + 1]
        
        # Strassen terms
        # Each term is a linear combination of A elements, B elements
        # U entries: for A elements
        # V entries: for B elements
        # W entries: for C elements
        
        # Term 0: M1 = A_top * B_top
        # A_top: i=0, j=a_cols[0], a_cols[1]
        # B_top: j=b_rows[0], k=b_cols[0], b_cols[1]
        # C_top: i=0, k=b_cols[0], b_cols[1]
        
        # For each of the 7 terms, define:
        # u_coeffs: dict from (i, j) to coefficient
        # v_coeffs: dict from (j, k) to coefficient
        # w_coeffs: dict from (i, k) to coefficient (with sign)
        
        # I'll define each term as (u_mults, v_mults, w_entries)
        # where u_mults is a list of (i, j, coeff), etc.
        
        # Actually, let me just hardcode the 7 terms for each block.
        
        # Term 0: M1
        # U: A[0, a_cols[0]] += 1, A[0, a_cols[1]] += 1
        # V: B[b_rows[0], b_cols[0]] += 1, B[b_rows[0], b_cols[1]] += 1
        # W: C[0, b_cols[0]] += 1, C[0, b_cols[1]] += 1
        
        # This is getting complex. Let me use a different approach.
        # For each Strassen term, I'll have one column in U, V, W.
        # U has one nonzero per row that's involved, V same, W same.
        
        # Let me define the 7 terms more carefully.
        
        # For a 2x2x2 Strassen:
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        
        # M1 = a00*b00 + a01*b01 -> contributes to c00, c01 with +1
        # M2 = a10*(b00+b10) + a11*(b01+b11) -> contributes to c00, c01 with +1
        # M3 = (a00+a10)*b10 + (a01+a11)*b11 -> contributes to c10, c11 with +1
        # M4 = a00*(b10-b00) + a01*(b11-b01) -> contributes to c10, c11 with +1
        # M5 = (a00-a10)*(b00+b10) + (a01-a11)*(b01+b11) -> contributes to c00,c01 with +1 and c10,c11 with +1
        # M6 = (a10-a00)*b00 + (a11-a01)*b01 -> contributes to c10, c11 with +1
        # M7 = (a00+a10)*(b10-b00) + (a01+a11)*(b11-b01) -> contributes to c00,c01 with -1
        
        # Wait, let me double-check the Strassen formula.
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Hmm, there are different conventions. Let me use a standard one.
        # Using the Wikipedia convention:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let me use this. For each term, I need:
        # U: linear combination of A elements
        # V: linear combination of B elements
        # W: contributes to C elements with certain signs
        
        # For each term, U has at most 2 nonzeros (one for each row of A that's involved),
        # V has at most 2 nonzeros, W has at most 4 nonzeros (for the 2x2 C).
        
        # Actually, for a single rank-1 term, U, V, W each have specific patterns.
        # Let me just construct it term by term.
        
        # I'll define each term as a tuple (u_entries, v_entries, w_entries)
        # where each entry is (index, coeff)
        
        # A row indices: 0 -> i=0, 1 -> i=1
        # A col indices: 0 -> j=a_cols[0], 1 -> j=a_cols[1]
        # B row indices: 0 -> j=b_rows[0], 1 -> j=b_rows[1]
        # B col indices: 0 -> k=b_cols[0], 1 -> k=b_cols[1]
        # C row indices: 0 -> i=0, 1 -> i=1
        # C col indices: 0 -> k=b_cols[0], 1 -> k=b_cols[1]
        
        # U row = i*4 + j
        # V row = j*4 + k
        # W row = k*2 + i
        
        # Term M1 = (A11+A22)(B11+B22)
        # U: A[0, a_cols[0]] + A[1, a_cols[1]] -> but that's not a single rank-1 term.
        # Actually, each M is a product of two vectors, so it's one rank-1 term.
        # M1 = (a00+a11)*(b00+b11) + (a01+a12)*(b01+b12) ... no.
        
        # I'm overcomplicating this. Let me just use the known rank-7 decomposition
        # for 2x2x2 and instantiate it for each of the 4 blocks.
        
        # For a 2x2x2, the 7 rank-1 terms can be written as:
        # t1: U=[1,0,0,0,1,0,0,0] (for A flattened), V=[1,0,1,0,0,1,0,1] (for B), W=[1,0,0,1,0,0,1,0] (for C)
        # This is getting too complex to do by hand.
        
        # Let me just use numpy to construct it numerically and then extract the integer entries.
        
        pass  # placeholder, will redo below
    
    # Let me restart with a cleaner approach.
    # I'll use the known Strassen 7-term decomposition for each 2x2x2 block.
    
    U = np.zeros((8, 28), dtype=float)
    V = np.zeros((16, 28), dtype=float)
    W = np.zeros((8, 28), dtype=float)
    
    # Strassen 7 terms for 2x2x2, using standard formula:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A @ B
    #
    # M1 = (a00+a11)(b00+b11) + (a01+a10)(b01+b10)  -- NO, that's not right.
    # Each Mi is a scalar, and C is a linear combination.
    #
    # Standard Strassen (7 scalar multiplications):
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B21 - B22)
    # M4 = A22 * (B12 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of two linear forms. So each is a rank-1 term.
    
    # For each block, I'll define the 7 terms.
    
    def add_strassen_block(a_blk, b_row_blk, b_col_blk, col_start):
        """Add 7 Strassen terms for one 2x2x2 block starting at col_start."""
        a_cols = [a_blk * 2, a_blk * 2 + 1]
        b_rows = [b_row_blk * 2, b_row_blk * 2 + 1]
        b_cols = [b_col_blk * 2, b_col_blk * 2 + 1]
        
        # Helper to get U, V, W indices
        def u_idx(i, j):
            return i * 4 + j
        
        def v_idx(j, k):
            return j * 4 + k
        
        def w_idx(i, k):
            return k * 2 + i
        
        # A indices: A[0, a_cols[0]], A[0, a_cols[1]], A[1, a_cols[0]], A[1, a_cols[1]]
        # B indices: B[b_rows[0], b_cols[0]], B[b_rows[0], b_cols[1]], B[b_rows[1], b_cols[0]], B[b_rows[1], b_cols[1]]
        # C indices: C[0, b_cols[0]], C[0, b_cols[1]], C[1, b_cols[0]], C[1, b_cols[1]]
        
        # Define the 7 terms
        # Each term: (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: list of (U_row, coeff)
        # v_coeffs: list of (V_row, coeff)
        # w_coeffs: list of (W_row, coeff)
        
        # M1 = (A11+A22)(B11+B22)
        # A11 = A[0, a_cols[0]], A22 = A[1, a_cols[1]]
        # B11 = B[b_rows[0], b_cols[0]], B22 = B[b_rows[1], b_cols[1]]
        # u: (u_idx(0, a_cols[0]), 1), (u_idx(1, a_cols[1]), 1)
        # v: (v_idx(b_rows[0], b_cols[0]), 1), (v_idx(b_rows[1], b_cols[1]), 1)
        # w: C11 += 1, C22 += 1
        # w: (w_idx(0, b_cols[0]), 1), (w_idx(1, b_cols[1]), 1)
        
        terms = [
            # M1
            ([(u_idx(0, a_cols[0]), 1), (u_idx(1, a_cols[1]), 1)],
             [(v_idx(b_rows[0], b_cols[0]), 1), (v_idx(b_rows[1], b_cols[1]), 1)],
             [(w_idx(0, b_cols[0]), 1), (w_idx(1, b_cols[1]), 1)]),
            # M2 = (A21+A22)*B11
            ([(u_idx(1, a_cols[0]), 1), (u_idx(1, a_cols[1]), 1)],
             [(v_idx(b_rows[0], b_cols[0]), 1)],
             [(w_idx(1, b_cols[0]), 1)]),
            # M3 = A11*(B21-B22)
            ([(u_idx(0, a_cols[0]), 1)],
             [(v_idx(b_rows[1], b_cols[0]), 1), (v_idx(b_rows[1], b_cols[1]), -1)],
             [(w_idx(0, b_cols[1]), 1)]),
            # M4 = A22*(B12-B11)
            ([(u_idx(1, a_cols[1]), 1)],
             [(v_idx(b_rows[0], b_cols[1]), 1), (v_idx(b_rows[0], b_cols[0]), -1)],
             [(w_idx(1, b_cols[1]), 1)]),
            # M5 = (A11+A12)*B22
            ([(u_idx(0, a_cols[0]), 1), (u_idx(0, a_cols[1]), 1)],
             [(v_idx(b_rows[1], b_cols[1]), 1)],
             [(w_idx(0, b_cols[1]), 1)]),
            # M6 = (A21-A11)*(B11+B12)
            ([(u_idx(1, a_cols[0]), 1), (u_idx(0, a_cols[0]), -1)],
             [(v_idx(b_rows[0], b_cols[0]), 1), (v_idx(b_rows[0], b_cols[1]), 1)],
             [(w_idx(1, b_cols[1]), 1)]),
            # M7 = (A12-A22)*(B21+B22)
            ([(u_idx(0, a_cols[1]), 1), (u_idx(1, a_cols[1]), -1)],
             [(v_idx(b_rows[1], b_cols[0]), 1), (v_idx(b_rows[1], b_cols[1]), 1)],
             [(w_idx(0, b_cols[0]), -1)]),
        ]
        
        for t, (u_c, v_c, w_c) in enumerate(terms):
            col = col_start + t
            for (idx, coeff) in u_c:
                U[idx, col] = coeff
            for (idx, coeff) in v_c:
                V[idx, col] = coeff
            for (idx, coeff) in w_c:
                W[idx, col] = coeff
    
    # Add the 4 blocks
    add_strassen_block(0, 0, 0, 0)   # A_L, B_LL -> C_L
    add_strassen_block(0, 0, 1, 7)   # A_L, B_LR -> C_R
    add_strassen_block(1, 1, 0, 14)  # A_R, B_RL -> C_L
    add_strassen_block(1, 1, 1, 21)  # A_R, B_RR -> C_R
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

