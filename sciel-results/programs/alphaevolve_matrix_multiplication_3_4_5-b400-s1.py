"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-minimal tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
              with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    rank = n * m * p      # 60 for schoolbook
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]  # 12 x 60
    V = [[0] * rank for _ in range(m * p)]  # 20 x 60
    W = [[0] * rank for _ in range(p * n)]  # 15 x 60
    
    # We will use two disjoint Strassen 2x2x2 blocks to save rank.
    # Block 1: A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] @ B[2:4, 2:4] -> C[0:2, 2:4]
    # Each Strassen block uses rank 7 instead of schoolbook rank 8, saving 1 each.
    # Total rank: 60 - 2 = 58
    
    # We'll build the decomposition term by term.
    # For non-Strassen terms, use schoolbook (1 term each).
    # For Strassen terms, use 7 terms per 2x2x2 block.
    
    # First, let's identify all the (i, j, k) tuples and assign them to either
    # Strassen blocks or schoolbook.
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    
    # All other (i, j, k) use schoolbook.
    
    # Let's build the Strassen decomposition for each block.
    # For a 2x2x2 block with local indices (ii, jj, kk) in {0,1}:
    # 
    # Strassen formulas (standard):
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For each Strassen term Mt, we need to add contributions to U, V, W.
    # Each Mt is a product of a linear combination of A entries and a linear combination of B entries.
    # We decompose each linear combination into individual terms.
    
    # Let's define the Strassen blocks explicitly.
    
    # Block 1: global indices i in {0,1}, j in {0,1}, k in {0,1}
    # Local to global: ii -> i = ii, jj -> j = jj, kk -> k = kk
    
    # Block 2: global indices i in {0,1}, j in {2,3}, k in {2,3}
    # Local to global: ii -> i = ii, jj -> j = 2 + jj, kk -> k = 2 + kk
    
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        """
        Add Strassen terms for a 2x2x2 block.
        i_offset, j_offset, k_offset: global offsets for local indices 0,1
        col_start: starting column index in U, V, W
        Returns: number of columns used (7)
        """
        # Define the 7 Strassen multipliers
        # Each multiplier is (coeff_A_list, coeff_B_list) where
        # coeff_A_list = [(ii, jj, coeff), ...] for A entries
        # coeff_B_list = [(jj, kk, coeff), ...] for B entries
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        M1_A = [((0, 0), 1), ((1, 1), 1)]
        M1_B = [((0, 0), 1), ((1, 1), 1)]
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        M2_A = [((1, 0), 1), ((1, 1), 1)]
        M2_B = [((0, 0), 1)]
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        M3_A = [((0, 0), 1)]
        M3_B = [((0, 1), 1), ((1, 1), -1)]
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        M4_A = [((1, 1), 1)]
        M4_B = [((1, 0), 1), ((0, 0), -1)]
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        M5_A = [((0, 0), 1), ((0, 1), 1)]
        M5_B = [((1, 1), 1)]
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        M6_A = [((1, 0), 1), ((0, 0), -1)]
        M6_B = [((0, 0), 1), ((0, 1), 1)]
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        M7_A = [((0, 1), 1), ((1, 1), -1)]
        M7_B = [((1, 0), 1), ((1, 1), 1)]
        
        multipliers = [
            (M1_A, M1_B),
            (M2_A, M2_B),
            (M3_A, M3_B),
            (M4_A, M4_B),
            (M5_A, M5_B),
            (M6_A, M6_B),
            (M7_A, M7_B),
        ]
        
        # For each multiplier, we need to add contributions to U, V, W for each (ii, kk) output.
        # C[ii, kk] gets contributions from various multipliers.
        
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # For each output (ii, kk), and for each multiplier that contributes to it,
        # we add a term to the decomposition.
        
        # Actually, let's think about this differently.
        # Each Strassen multiplier Mt is a scalar that we compute.
        # The output C[ii, kk] is a linear combination of the Mt's.
        # 
        # In the tensor decomposition, each "column" c corresponds to one scalar product
        # U[:, c] * V[:, c] that contributes to W.
        # 
        # For Strassen, we have 7 scalar products (the Mt's).
        # Each Mt = (sum of A terms) * (sum of B terms)
        # We can expand this into individual terms, but that would give us more than 7.
        # 
        # The key insight is that in the tensor decomposition, we can have
        # U[i*m+j, c] and V[j*p+k, c] that are linear combinations, and
        # W[k*n+i, c] that collects them.
        # 
        # Actually, for each Mt, we can assign one column c.
        # U[:, c] has entries that are the coefficients from the A linear combination.
        # V[:, c] has entries that are the coefficients from the B linear combination.
        # W[:, c] has entries that indicate which output (ii, kk) this Mt contributes to.
        
        col = col_start
        for m_idx, (A_terms, B_terms) in enumerate(multipliers):
            # A_terms: list of ((ii, jj), coeff)
            # B_terms: list of ((jj, kk), coeff)
            
            # For U: index is i*m + j
            for (ii, jj), coeff_A in A_terms:
                i_global = i_offset + ii
                j_global = j_offset + jj
                U[i_global * m + j_global][col] += coeff_A
            
            # For V: index is j*p + k
            for (jj, kk), coeff_B in B_terms:
                j_global = j_offset + jj
                k_global = k_offset + kk
                V[j_global * p + k_global][col] += coeff_B
            
            # For W: we need to add contributions for each output (ii, kk)
            # that this multiplier affects.
            # We need to figure out which (ii, kk) this multiplier contributes to.
            
            # This is tricky because a single multiplier can contribute to multiple outputs.
            # In the standard Strassen, each Mt is used in one or more output formulas.
            # 
            # Let's hardcode the contributions for each multiplier.
            
            # M1 (m_idx=0): contributes to C[0,0] with +1, C[1,1] with +1
            if m_idx == 0:  # M1
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1  # C[0,0]
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1  # C[1,1]
            
            # M2 (m_idx=1): contributes to C[1,0] with +1, C[1,1] with -1
            elif m_idx == 1:  # M2
                W[(k_offset + 0) * n + (i_offset + 1)][col] += 1  # C[1,0]
                W[(k_offset + 1) * n + (i_offset + 1)][col] += -1  # C[1,1]
            
            # M3 (m_idx=2): contributes to C[0,1] with +1, C[1,1] with +1
            elif m_idx == 2:  # M3
                W[(k_offset + 1) * n + (i_offset + 0)][col] += 1  # C[0,1]
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1  # C[1,1]
            
            # M4 (m_idx=3): contributes to C[0,0] with +1, C[1,0] with +1
            elif m_idx == 3:  # M4
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1  # C[0,0]
                W[(k_offset + 0) * n + (i_offset + 1)][col] += 1  # C[1,0]
            
            # M5 (m_idx=4): contributes to C[0,0] with -1, C[0,1] with +1
            elif m_idx == 4:  # M5
                W[(k_offset + 0) * n + (i_offset + 0)][col] += -1  # C[0,0]
                W[(k_offset + 1) * n + (i_offset + 0)][col] += 1  # C[0,1]
            
            # M6 (m_idx=5): contributes to C[1,1] with +1
            elif m_idx == 5:  # M6
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1  # C[1,1]
            
            # M7 (m_idx=6): contributes to C[0,0] with +1
            elif m_idx == 6:  # M7
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1  # C[0,0]
            
            col += 1
        
        return col - col_start
    
    # Add Strassen block 1: i_offset=0, j_offset=0, k_offset=0
    col = 0
    col += add_strassen_block(0, 0, 0, col)  # Uses columns 0-6
    
    # Add Strassen block 2: i_offset=0, j_offset=2, k_offset=2
    col += add_strassen_block(0, 2, 2, col)  # Uses columns 7-13
    
    # Now add schoolbook terms for all other (i, j, k) tuples.
    # We need to skip the ones already covered by Strassen blocks.
    # Block 1 covers: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2 covers: i in {0,1}, j in {2,3}, k in {2,3}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i, j, k) is covered by a Strassen block
                covered = False
                # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    covered = True
                # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
                if i in (0, 1) and j in (2, 3) and k in (2, 3):
                    covered = True
                
                if not covered:
                    # Schoolbook: one term
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # The total rank should be 58 (7 + 7 + 46 schoolbook terms)
    # Schoolbook terms: total 60 - 8 (block 1) - 8 (block 2) = 44
    # Wait, let me recount.
    # Block 1: 2*2*2 = 8 terms
    # Block 2: 2*2*2 = 8 terms
    # Total covered: 16
    # Schoolbook: 60 - 16 = 44
    # Strassen: 7 + 7 = 14
    # Total: 14 + 44 = 58
    
    # Trim U, V, W to the actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
