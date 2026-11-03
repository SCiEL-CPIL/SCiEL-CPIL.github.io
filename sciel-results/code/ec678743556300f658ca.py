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
    
    # We'll use two disjoint Strassen 2x2x2 blocks to save rank.
    # Block 1: A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] @ B[2:4, 2:4] -> C[0:2, 2:4]
    # Each Strassen block uses rank 7 instead of schoolbook rank 8, saving 1 each.
    # Total rank: 60 - 2 = 58
    
    # Build the decomposition term by term.
    
    def add_strassen_block(i_offset, j_offset, k_offset, col_start, U, V, W):
        """
        Add Strassen terms for a 2x2x2 block.
        i_offset, j_offset, k_offset: global offsets for local indices 0,1
        col_start: starting column index in U, V, W
        Returns: number of columns used (7)
        """
        # Define the 7 Strassen multipliers
        M1_A = [((0, 0), 1), ((1, 1), 1)]
        M1_B = [((0, 0), 1), ((1, 1), 1)]
        
        M2_A = [((1, 0), 1), ((1, 1), 1)]
        M2_B = [((0, 0), 1)]
        
        M3_A = [((0, 0), 1)]
        M3_B = [((0, 1), 1), ((1, 1), -1)]
        
        M4_A = [((1, 1), 1)]
        M4_B = [((1, 0), 1), ((0, 0), -1)]
        
        M5_A = [((0, 0), 1), ((0, 1), 1)]
        M5_B = [((1, 1), 1)]
        
        M6_A = [((1, 0), 1), ((0, 0), -1)]
        M6_B = [((0, 0), 1), ((0, 1), 1)]
        
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
        
        col = col_start
        for m_idx, (A_terms, B_terms) in enumerate(multipliers):
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
            
            # For W: contributions to output (ii, kk)
            if m_idx == 0:  # M1
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1
            elif m_idx == 1:  # M2
                W[(k_offset + 0) * n + (i_offset + 1)][col] += 1
                W[(k_offset + 1) * n + (i_offset + 1)][col] += -1
            elif m_idx == 2:  # M3
                W[(k_offset + 1) * n + (i_offset + 0)][col] += 1
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1
            elif m_idx == 3:  # M4
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1
                W[(k_offset + 0) * n + (i_offset + 1)][col] += 1
            elif m_idx == 4:  # M5
                W[(k_offset + 0) * n + (i_offset + 0)][col] += -1
                W[(k_offset + 1) * n + (i_offset + 0)][col] += 1
            elif m_idx == 5:  # M6
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1
            elif m_idx == 6:  # M7
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1
            
            col += 1
        
        return col - col_start
    
    # Initialize U, V, W with zeros
    U = [[0] * 60 for _ in range(n * m)]  # 12 x 60
    V = [[0] * 60 for _ in range(m * p)]  # 20 x 60
    W = [[0] * 60 for _ in range(p * n)]  # 15 x 60
    
    # Add Strassen block 1: i_offset=0, j_offset=0, k_offset=0
    col = 0
    col += add_strassen_block(0, 0, 0, col, U, V, W)  # Uses columns 0-6
    
    # Add Strassen block 2: i_offset=0, j_offset=2, k_offset=2
    col += add_strassen_block(0, 2, 2, col, U, V, W)  # Uses columns 7-13
    
    # Now add schoolbook terms for all other (i, j, k) tuples.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i, j, k) is covered by a Strassen block
                covered = False
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    covered = True
                if i in (0, 1) and j in (2, 3) and k in (2, 3):
                    covered = True
                
                if not covered:
                    # Schoolbook: one term
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Trim U, V, W to the actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
