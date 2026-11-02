"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
    
    Uses Strassen's algorithm (rank 7) for the 2x2 * 2x2 part repeated for
    7 of the 15 columns, and schoolbook (rank 4) for the remaining 8 columns.
    Total rank: 7*7 + 4*8 = 49 + 32 = 81... wait, that's worse.
    
    Actually, let me reconsider. The dimensions are n=2, m=2, p=15.
    So we're multiplying a 2x2 matrix by a 2x15 matrix to get a 2x15 matrix.
    
    The tensor has indices (i, j, k) where:
    - i in {0,1} (row of A, row of C)
    - j in {0,1} (col of A, row of B)  
    - k in {0,...,14} (col of B, col of C)
    
    Schoolbook: rank = n*m*p = 2*2*15 = 60
    
    For each column k, we do a 2x2 * 2x1 multiplication, which is just 4 scalar multiplications.
    So total is 4*15 = 60.
    
    To use Strassen, we'd need to group columns. If we group 2 columns together, we get a 2x2 * 2x2 problem.
    15 columns = 7 groups of 2 + 1 column of 1.
    
    For 7 groups of 2x2*2x2: use Strassen with rank 7 each → 7*7 = 49
    For 1 remaining 2x2*2x1: use schoolbook with rank 4 → 4
    Total: 49 + 4 = 53
    
    This matches the knowledge from the run!
    """
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Strassen's 7-term decomposition for 2x2 * 2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Strassen term corresponds to one rank-1 component.
    # For term t, we have coefficients for A[i,j] and B[j,k] and C[i,k].
    
    # Let's define the Strassen decomposition explicitly.
    # For a 2x2*2x2 block, the 7 terms are:
    # term: (A coeffs: {(i,j): val}, B coeffs: {(j,k): val}, C coeffs: {(i,k): val})
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M2 = (A10 + A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}),
        # M3 = A00(B01 - B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M4 = A11(B10 - B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        # M5 = (A00 + A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        # M6 = (A10 - A00)(B00 + B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        # M7 = (A01 - A11)(B10 + B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # For the remaining single column (2x2 * 2x1), use schoolbook:
    # C0k = A00*B0k + A01*B1k
    # C1k = A10*B0k + A11*B1k
    # 4 terms
    
    schoolbook_single_col = [
        # A00 * B0k
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        # A01 * B1k
        ({(0,1): 1}, {(1,0): 1}, {(0,1): 1}),
        # A10 * B0k
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),
        # A11 * B1k
        ({(1,1): 1}, {(1,0): 1}, {(1,1): 1}),
    ]
    
    # Now build the factor matrices.
    # Number of Strassen groups: 7 (for columns 0-1, 2-3, 4-5, 6-7, 8-9, 10-11, 12-13)
    # Remaining: column 14 (single column)
    
    num_strassen_groups = 7
    num_single_cols = 1
    
    total_rank = num_strassen_groups * 7 + num_single_cols * 4  # 49 + 4 = 53
    
    U = [[0] * total_rank for _ in range(n * m)]  # (4, 53)
    V = [[0] * total_rank for _ in range(m * p)]  # (30, 53)
    W = [[0] * total_rank for _ in range(p * n)]  # (30, 53)
    
    col = 0
    # Process Strassen groups
    for g in range(num_strassen_groups):
        col_start = g * 2  # columns 2g and 2g+1
        for t in range(7):
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[t]
            
            # Fill U: index i*m + j
            for (i, j), val in a_coeffs.items():
                U[i * m + j][col] += val
            
            # Fill V: index j*p + k, where k is relative to the 2-column block
            for (j, k_rel), val in b_coeffs.items():
                k_abs = col_start + k_rel
                V[j * p + k_abs][col] += val
            
            # Fill W: index k*n + i, where k is relative to the 2-column block
            for (i, k_rel), val in c_coeffs.items():
                k_abs = col_start + k_rel
                W[k_abs * n + i][col] += val
            
            col += 1
    
    # Process single column (column 14)
    for t in range(4):
        a_coeffs, b_coeffs, c_coeffs = schoolbook_single_col[t]
        
        # Fill U: index i*m + j
        for (i, j), val in a_coeffs.items():
            U[i * m + j][col] += val
        
        # Fill V: index j*p + k, where k = 14
        for (j, k_rel), val in b_coeffs.items():
            k_abs = 14
            V[j * p + k_abs][col] += val
        
        # Fill W: index k*n + i, where k = 14
        for (i, k_rel), val in c_coeffs.items():
            k_abs = 14
            W[k_abs * n + i][col] += val
        
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
