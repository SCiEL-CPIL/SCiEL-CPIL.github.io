# EVOLVE-BLOCK-START
import time

def construct():
    """
    Rank-28 decomposition for (2x4)*(4x4) matrix multiplication.
    
    Strategy: Partition the 4x4 matrix B into four 2x2 blocks.
    Partition the 2x4 matrix A into two 2x2 blocks (left and right).
    The product consists of 4 independent 2x2 matrix multiplications
    (A_left * B_TL + A_right * B_BL, etc.).
    Each 2x2 multiplication uses Strassen's rank-7 algorithm.
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # We will construct U, V, W matrices.
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Indices for U: i*m + j, i in {0,1}, j in {0,1,2,3}
    # Indices for V: j*p + k, j in {0,1,2,3}, k in {0,1,2,3}
    # Indices for W: k*n + i, k in {0,1,2,3}, i in {0,1}
    
    # Strassen coefficients for 2x2 multiplication
    # Inputs: A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # Output C:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # We have 4 block products:
    # 1. A_L * B_TL -> contributes to C[:, 0:2]
    # 2. A_R * B_BL -> contributes to C[:, 0:2]
    # 3. A_L * B_TR -> contributes to C[:, 2:4]
    # 4. A_R * B_BR -> contributes to C[:, 2:4]
    
    # Let's define the 4 products explicitly.
    # Product 1: A_L (cols 0,1 of A) * B_TL (rows 0,1; cols 0,1 of B)
    # Product 2: A_R (cols 2,3 of A) * B_BL (rows 2,3; cols 0,1 of B)
    # Product 3: A_L (cols 0,1 of A) * B_TR (rows 0,1; cols 2,3 of B)
    # Product 4: A_R (cols 2,3 of A) * B_BR (rows 2,3; cols 2,3 of B)
    
    # For each product, we add 7 terms to U, V, W.
    
    def add_strassen_term(term_idx, a_indices, b_indices, c_indices, coeffs):
        """
        term_idx: 0-6 for M1-M7
        a_indices: list of (i, j) in A, with coefficient
        b_indices: list of (j, k) in B, with coefficient
        c_indices: list of (i, k) in C, with coefficient
        """
        col = term_idx
        # This helper is too complex for inline use, I'll just hardcode the loops.
        pass

    # Instead of a helper, I will iterate through the 4 blocks and 7 Strassen terms.
    
    blocks = [
        # (A_col_start, B_row_start, B_col_start, C_col_start)
        (0, 0, 0, 0), # A_L, B_TL -> C_L
        (2, 2, 0, 0), # A_R, B_BL -> C_L
        (0, 0, 2, 2), # A_L, B_TR -> C_R
        (2, 2, 2, 2), # A_R, B_BR -> C_R
    ]
    
    # Strassen definitions
    # Each term is defined by linear combinations of A entries, B entries, and C entries.
    # A entries are indexed by (i, j) where i in {0,1}, j in {0,1} relative to block.
    # B entries are indexed by (j, k) where j in {0,1}, k in {0,1} relative to block.
    # C entries are indexed by (i, k) where i in {0,1}, k in {0,1} relative to block.
    
    # M1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    # M2: (a10+a11)b00 -> c10 += 1
    # M3: a00(b01-b11) -> c01 += 1
    # M4: a11(b10-b00) -> c00 += 1, c10 += 1
    # M5: (a00+a01)b11 -> c00 -= 1, c01 += 1
    # M6: (a10-a00)(b00+b01) -> c11 += 1
    # M7: (a01-a11)(b10+b11) -> c00 += 1
    
    # Let's map relative indices to global indices.
    # Global A index: i * 4 + (a_col_start + j_rel)
    # Global B index: (b_row_start + j_rel) * 4 + (b_col_start + k_rel)
    # Global C index: (c_col_start + k_rel) * 2 + i
    
    strassen_terms = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: dict { (i_rel, j_rel): coeff }
        # b_coeffs: dict { (j_rel, k_rel): coeff }
        # c_coeffs: dict { (i_rel, k_rel): coeff }
        
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for b_idx, (a_cs, b_rs, b_cs, c_cs) in enumerate(blocks):
        for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(strassen_terms):
            col = b_idx * 7 + t_idx
            
            # Fill U
            for (i_rel, j_rel), coeff in a_coeff.items():
                i_glob = i_rel
                j_glob = a_cs + j_rel
                U[i_glob * 4 + j_glob][col] += coeff
                
            # Fill V
            for (j_rel, k_rel), coeff in b_coeff.items():
                j_glob = b_rs + j_rel
                k_glob = b_cs + k_rel
                V[j_glob * 4 + k_glob][col] += coeff
                
            # Fill W
            for (i_rel, k_rel), coeff in c_coeff.items():
                i_glob = i_rel
                k_glob = c_cs + k_rel
                W[k_glob * 2 + i_glob][col] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
