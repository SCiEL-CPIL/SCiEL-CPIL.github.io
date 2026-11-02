import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor M_{2,4,4}.
    Uses a block Strassen decomposition.
    Splits the 2x4 x 4x4 multiplication into four 2x2 x 2x2 multiplications.
    Each 2x2 multiplication is decomposed using Strassen's algorithm (rank 7).
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS
    
    rank = 28
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index mapping:
    # U rows: i * m + j (i in 0..1, j in 0..3)
    # V rows: j * p + k (j in 0..3, k in 0..3)
    # W rows: k * n + i (k in 0..3, i in 0..1)
    
    # We define 4 blocks of 2x2 multiplications:
    # Block 1: A_L * B_11 -> C_L (top-left 2x2 of C)
    # Block 2: A_R * B_21 -> C_L (top-left 2x2 of C)
    # Block 3: A_L * B_12 -> C_R (top-right 2x2 of C)
    # Block 4: A_R * B_22 -> C_R (top-right 2x2 of C)
    
    # A_L indices: i in {0,1}, j in {0,1}
    # A_R indices: i in {0,1}, j in {2,3}
    # B_11 indices: j in {0,1}, k in {0,1}
    # B_21 indices: j in {2,3}, k in {0,1}
    # B_12 indices: j in {0,1}, k in {2,3}
    # B_22 indices: j in {2,3}, k in {2,3}
    
    # Strassen coefficients for 2x2 multiplication:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    # We iterate over the 4 blocks and the 7 Strassen terms.
    
    # Block definitions: (A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_col_offset)
    # A is 2x4. B is 4x4. C is 2x4.
    # A_L: rows 0-1, cols 0-1. A_R: rows 0-1, cols 2-3.
    # B_11: rows 0-1, cols 0-1. B_21: rows 2-3, cols 0-1.
    # B_12: rows 0-1, cols 2-3. B_22: rows 2-3, cols 2-3.
    # C_L: cols 0-1. C_R: cols 2-3.
    
    blocks = [
        # (A_i_base, A_j_base, B_j_base, B_k_base, C_k_base)
        (0, 0, 0, 0, 0), # A_L * B_11 -> C_L
        (0, 2, 2, 0, 0), # A_R * B_21 -> C_L
        (0, 0, 0, 2, 2), # A_L * B_12 -> C_R
        (0, 2, 2, 2, 2), # A_R * B_22 -> C_R
    ]
    
    # Strassen term definitions:
    # Each term is (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict mapping (r,c) to coeff
    # b_coeffs: dict mapping (r,c) to coeff
    # c_coeffs: dict mapping (r,c) to coeff
    
    strassen_terms = [
        # M1
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1} # Wait, C00 and C11 get M1.
            # C00 = M1 - M4 + M6 - M7
            # C11 = M1 - M3 + M2 - M5
            # So M1 contributes to C00 and C11.
        ),
        # M2
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0} # C10 gets M2.
        ),
        # M3
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0} # C01 gets M3.
        ),
        # M4
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): -1, (0,1): 0, (1,0): 1, (1,1): 0} # C00 gets -M4, C10 gets M4.
        ),
        # M5
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0} # C01 gets M5, C11 gets -M5.
            # Wait, C11 = M1 - M3 + M2 - M5. So M5 contributes -1 to C11.
            # C01 = M3 + M5. So M5 contributes 1 to C01.
            # My c_coeffs dict should reflect the contribution to the local 2x2 C.
            # So for M5: C01 += 1, C11 += -1.
        ),
        # M6
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0} # C00 gets M6.
        ),
        # M7
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): -1, (0,1): 0, (1,0): 0, (1,1): 0} # C00 gets -M7.
        ),
    ]
    
    # Fix c_coeffs for M1, M5
    # M1: C00 += 1, C11 += 1
    # M5: C01 += 1, C11 += -1
    
    strassen_terms[0] = (
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}
    )
    
    strassen_terms[4] = (
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): -1}
    )
    
    col_idx = 0
    for block in blocks:
        A_i_base, A_j_base, B_j_base, B_k_base, C_k_base = block
        
        for term in strassen_terms:
            a_coeffs, b_coeffs, c_coeffs = term
            
            # Fill U
            for (ar, ac), val in a_coeffs.items():
                u_row = A_i_base + ar # i index
                # Wait, A is 2x4. i is row, j is col.
                # A_j_base is the column offset in A.
                # So global col j = A_j_base + ac.
                # U row index: i * m + j = (A_i_base + ar) * 4 + (A_j_base + ac)
                u_idx = (A_i_base + ar) * m + (A_j_base + ac)
                U[u_idx, col_idx] += val
            
            # Fill V
            for (br, bc), val in b_coeffs.items():
                # B is 4x4. j is row, k is col.
                # B_j_base is row offset. B_k_base is col offset.
                # Global row j = B_j_base + br.
                # Global col k = B_k_base + bc.
                # V row index: j * p + k = (B_j_base + br) * 4 + (B_k_base + bc)
                v_idx = (B_j_base + br) * p + (B_k_base + bc)
                V[v_idx, col_idx] += val
            
            # Fill W
            for (cr, cc), val in c_coeffs.items():
                # C is 2x4. i is row, k is col.
                # C_k_base is col offset.
                # Global row i = cr (since A has 2 rows, and C has 2 rows, no offset for i).
                # Wait, C row index corresponds to A row index.
                # In the block, cr is 0 or 1.
                # So global i = cr.
                # Global col k = C_k_base + cc.
                # W row index: k * n + i = (C_k_base + cc) * 2 + cr
                w_idx = (C_k_base + cc) * n + cr
                W[w_idx, col_idx] += val
            
            col_idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
