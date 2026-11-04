import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 6x6 = 2x2 blocks of 3x3
    # Use Strassen's 7-term algorithm at block level
    # Each block multiplication uses schoolbook 3x3 (rank 27)
    # Total rank = 7 * 27 = 189
    
    total_rank = 189
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Strassen's 7 products for 2x2 block multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each product Pk is a 3x3 matrix multiplication
    # For each Pk, we need rank-27 decomposition for 3x3
    
    # Strassen terms: (A_blocks with coeffs, B_blocks with coeffs, C_contributions)
    # Block indices: 0=top-left, 1=bottom-right for 2x2 grid
    strassen_terms = [
        # P1 = (A00 + A11) * (B00 + B11)
        {
            'A': [(0, 1), (3, 1)],   # A00 at local (0,0), A11 at local (3,3) -> rows 0-2, cols 0-2 and rows 3-5, cols 3-5
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]     # contributes to C00 and C11
        },
        # P2 = (A10 + A11) * B00
        {
            'A': [(3, 0), (3, 3)],    # A10 at rows 3-5, cols 0-2; A11 at rows 3-5, cols 3-5
            'B': [(0, 1)],
            'C': [(3, 0), (3, 3)]     # contributes to C10 and C11
        },
        # P3 = A00 * (B01 - B11)
        {
            'A': [(0, 1)],
            'B': [(0, 3), (3, -1)],   # B01 at cols 3-5, B11 at cols 3-5 with -1
            'C': [(0, 3), (3, 3)]     # contributes to C01 and C11
        },
        # P4 = A11 * (B10 - B00)
        {
            'A': [(3, 3)],
            'B': [(3, 0), (0, -1)],   # B10 at rows 3-5, cols 0-2; B00 at rows 0-2, cols 0-2 with -1
            'C': [(0, 0), (3, 0)]     # contributes to C00 and C10
        },
        # P5 = (A00 + A01) * B11
        {
            'A': [(0, 0), (0, 3)],    # A00 at cols 0-2, A01 at cols 3-5
            'B': [(3, 3)],
            'C': [(0, -1), (0, 3)]    # contributes to C00 with -1, C01 with +1
        },
        # P6 = (A10 - A00) * (B00 + B01)
        {
            'A': [(3, 0), (0, -1)],   # A10 with +1, A00 with -1
            'B': [(0, 0), (0, 3)],    # B00 and B01
            'C': [(3, 3)]             # contributes to C11
        },
        # P7 = (A01 - A11) * (B10 + B11)
        {
            'A': [(0, 3), (3, -1)],   # A01 with +1, A11 with -1
            'B': [(3, 0), (3, 3)],    # B10 and B11
            'C': [(0, 0)]             # contributes to C00
        },
    ]
    
    # For each Strassen term, build the rank-27 3x3 decomposition
    col_offset = 0
    for term in strassen_terms:
        # For this Strassen term, we compute a 3x3 * 3x3 = 3x3 multiplication
        # The A part is a linear combination of 3x3 blocks
        # The B part is a linear combination of 3x3 blocks
        # The result contributes to C blocks
        
        # Build local 3x3 factors for this term
        # U_local[i, j] gets contributions from each A block
        # V_local[j, k] gets contributions from each B block
        # W_local[k, i] gets contributions for each C block
        
        U_local = np.zeros((9, 27))
        V_local = np.zeros((9, 27))
        W_local = np.zeros((9, 27))
        
        # Schoolbook 3x3: for each (i, j, k) in 0..2
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U_local[i * 3 + j, r] = 1.0
                    V_local[j * 3 + k, r] = 1.0
                    W_local[k * 3 + i, r] = 1.0
        
        # Now map to global indices
        # For each A block contribution (row_block, coeff):
        #   local row i, col j -> global row (row_block*3 + i)*6 + (col_block*3 + j)
        # But we need to track which col_block each A block uses
        
        # Let me restructure: for each term, explicitly list the A blocks with their (row_block, col_block, coeff)
        # and similarly for B and C
        
        # Redefine terms more explicitly
        pass
    
    # Let me redo this more carefully with explicit block positions
    
    # Clear and rebuild
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Explicit Strassen terms with full block specifications
    # Each term: list of (A_row_block, A_col_block, A_coeff), (B_row_block, B_col_block, B_coeff), (C_row_block, C_col_block, C_coeff)
    
    # P1 = (A[0:3,0:3] + A[3:6,3:6]) * (B[0:3,0:3] + B[3:6,3:6])
    #     contributes to C[0:3,0:3] and C[3:6,3:6]
    strassen_explicit = [
        # P1
        {
            'A': [((0, 0), 1), ((1, 1), 1)],
            'B': [((0, 0), 1), ((1, 1), 1)],
            'C': [((0, 0), 1), ((1, 1), 1)]
        },
        # P2 = (A[3:6,0:3] + A[3:6,3:6]) * B[0:3,0:3]
        #     contributes to C[3:6,0:3] and C[3:6,3:6]
        {
            'A': [((1, 0), 1), ((1, 1), 1)],
            'B': [((0, 0), 1)],
            'C': [((1, 0), 1), ((1, 1), 1)]
        },
        # P3 = A[0:3,0:3] * (B[0:3,3:6] - B[3:6,3:6])
        #     contributes to C[0:3,3:6] and C[3:6,3:6]
        {
            'A': [((0, 0), 1)],
            'B': [((0, 1), 1), ((1, 1), -1)],
            'C': [((0, 1), 1), ((1, 1), 1)]
        },
        # P4 = A[3:6,3:6] * (B[3:6,0:3] - B[0:3,0:3])
        #     contributes to C[0:3,0:3] and C[3:6,0:3]
        {
            'A': [((1, 1), 1)],
            'B': [((1, 0), 1), ((0, 0), -1)],
            'C': [((0, 0), 1), ((1, 0), 1)]
        },
        # P5 = (A[0:3,0:3] + A[0:3,3:6]) * B[3:6,3:6]
        #     contributes to C[0:3,0:3] with -1 and C[0:3,3:6] with +1
        {
            'A': [((0, 0), 1), ((0, 1), 1)],
            'B': [((1, 1), 1)],
            'C': [((0, 0), -1), ((0, 1), 1)]
        },
        # P6 = (A[3:6,0:3] - A[0:3,0:3]) * (B[0:3,0:3] + B[0:3,3:6])
        #     contributes to C[3:6,3:6]
        {
            'A': [((1, 0), 1), ((0, 0), -1)],
            'B': [((0, 0), 1), ((0, 1), 1)],
            'C': [((1, 1), 1)]
        },
        # P7 = (A[0:3,3:6] - A[3:6,3:6]) * (B[3:6,0:3] + B[3:6,3:6])
        #     contributes to C[0:3,0:3]
        {
            'A': [((0, 1), 1), ((1, 1), -1)],
            'B': [((1, 0), 1), ((1, 1), 1)],
            'C': [((0, 0), 1)]
        },
    ]
    
    col_idx = 0
    for term in strassen_explicit:
        A_blocks = term['A']
        B_blocks = term['B']
        C_blocks = term['C']
        
        # For each local (i', j', k') in 3x3 schoolbook
        for i_prime in range(3):
            for j_prime in range(3):
                for k_prime in range(3):
                    # U contributions
                    for (ar, ac), a_coeff in A_blocks:
                        i_global = ar * 3 + i_prime
                        j_global = ac * 3 + j_prime
                        u_row = i_global * 6 + j_global
                        U[u_row, col_idx] += a_coeff
                    
                    # V contributions
                    for (br, bc), b_coeff in B_blocks:
                        j_global = br * 3 + j_prime
                        k_global = bc * 3 + k_prime
                        v_row = j_global * 6 + k_global
                        V[v_row, col_idx] += b_coeff
                    
                    # W contributions
                    for (cr, cc), c_coeff in C_blocks:
                        i_global = cr * 3 + i_prime
                        k_global = cc * 3 + k_prime
                        w_row = k_global * 6 + i_global  # trace convention: W[k*n + i]
                        W[w_row, col_idx] += c_coeff
                    
                    col_idx += 1
    
    # Verify we used all columns
    assert col_idx == total_rank, f"Expected {total_rank} columns, got {col_idx}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
