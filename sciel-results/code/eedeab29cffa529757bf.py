import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Laderman coefficients for 3x3 matrix multiplication
    # Each term is (X_indices, Y_indices) where indices are (i, j, coeff)
    # X is from A, Y is from B
    # Block index (i, j) corresponds to A[i, j]
    
    laderman_terms = [
        # P0
        ([(0, 0, 1)], [(1, 1, 1), (2, 1, -1)]),
        # P1
        ([(0, 1, 1), (0, 0, -1)], [(0, 0, 1)]),
        # P2
        ([(1, 0, 1), (2, 0, -1)], [(2, 2, 1)]),
        # P3
        ([(2, 2, 1)], [(0, 0, 1), (0, 1, -1)]),
        # P4
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # P5
        ([(2, 2, 1), (1, 1, 1)], [(2, 2, 1), (1, 1, 1)]),
        # P6
        ([(0, 2, 1), (2, 2, 1)], [(2, 0, 1), (0, 0, 1)]),
        # P7
        ([(2, 0, 1), (0, 0, 1)], [(0, 2, 1), (2, 2, 1)]),
        # P8
        ([(0, 1, 1), (1, 1, 1)], [(1, 0, 1), (0, 0, 1)]),
        # P9
        ([(1, 0, 1), (0, 0, 1)], [(0, 1, 1), (1, 1, 1)]),
        # P10
        ([(1, 2, 1), (2, 2, 1)], [(2, 1, 1), (1, 1, 1)]),
        # P11
        ([(2, 1, 1), (1, 1, 1)], [(1, 2, 1), (2, 2, 1)]),
        # P12
        ([(0, 2, 1), (1, 2, 1)], [(1, 0, 1), (2, 0, 1)]),
        # P13
        ([(1, 0, 1), (0, 2, 1)], [(2, 1, 1), (0, 1, 1)]),
        # P14
        ([(2, 1, 1), (0, 1, 1)], [(0, 2, 1), (1, 2, 1)]),
        # P15
        ([(0, 2, 1), (1, 2, -1), (2, 2, 1)], [(1, 0, 1), (2, 0, -1), (0, 0, 1)]),
        # P16
        ([(2, 0, 1), (0, 2, -1), (0, 0, 1)], [(0, 1, 1), (1, 1, -1), (2, 2, 1)]),
        # P17
        ([(1, 0, 1), (2, 0, 1), (0, 0, -1)], [(0, 1, 1), (1, 1, 1), (2, 2, -1)]),
        # P18
        ([(2, 1, 1), (0, 1, 1), (1, 1, -1)], [(1, 2, 1), (2, 2, 1), (0, 2, -1)]),
        # P19
        ([(0, 2, 1), (1, 2, 1), (2, 2, -1)], [(2, 0, 1), (0, 0, 1), (1, 0, -1)]),
        # P20
        ([(0, 0, 1), (1, 1, -1), (2, 2, 1)], [(1, 1, 1), (0, 0, -1), (2, 2, 1)]),
        # P21
        ([(1, 1, 1), (2, 2, -1), (0, 0, 1)], [(2, 2, 1), (1, 1, -1), (0, 0, 1)]),
        # P22
        ([(2, 2, 1), (0, 0, -1), (1, 1, 1)], [(0, 0, 1), (2, 2, -1), (1, 1, 1)]),
    ]

    # Strassen coefficients for 2x2 matrix multiplication
    # Returns list of (x_indices, y_indices) for the 7 products
    # x_indices: [(p, q, coeff)] for X[p,q]
    # y_indices: [(p, q, coeff)] for Y[p,q]
    def get_strassen_factors():
        return [
            # S1
            ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
            # S2
            ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
            # S3
            ([(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
            # S4
            ([(1, 1, 1)], [(0, 1, 1), (0, 0, -1)]),
            # S5
            ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
            # S6
            ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
            # S7
            ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
        ]

    strassen_factors = get_strassen_factors()
    
    # Reconstruction of P_t from S_{t,s}
    # P_t[0,0] = S1 + S4 - S5 + S7
    # P_t[0,1] = S3 + S5
    # P_t[1,0] = S2 + S4
    # P_t[1,1] = S1 - S2 + S3 + S6
    strassen_recon = [
        [1, 0, 0, 1, -1, 0, 1], # P00
        [0, 0, 1, 0, 1, 0, 0],  # P01
        [0, 1, 0, 1, 0, 0, 0],  # P10
        [1, -1, 1, 0, 0, 1, 0]  # P11
    ]
    
    # Reconstruction of C_ij from P_t
    # This is the tricky part. We need the Laderman reconstruction coefficients.
    # C_ij is a 2x2 block.
    # C_ij = sum_t d[t, ij] P_t ? No, P_t are 2x2 matrices.
    # C_ij = sum_t (sum_{ab} d[t, ij, ab] P_t[ab]) ?
    # Actually, C_ij is a block. The entries of C_ij are linear combinations of entries of P_t.
    # Let's use the known Laderman reconstruction:
    # C_11 = P_1 + P_4 - P_8 + P_9
    # C_12 = P_2 + P_5 - P_9 + P_10
    # C_13 = P_3 + P_6 - P_10 + P_11
    # C_21 = P_4 + P_7 - P_12 + P_13
    # C_22 = P_5 + P_8 - P_13 + P_14
    # C_23 = P_6 + P_9 - P_14 + P_15
    # C_31 = P_7 + P_{10} - P_{15} + P_{16}
    # C_32 = P_8 + P_{11} - P_{16} + P_{17}
    # C_33 = P_9 + P_{12} - P_{17} + P_{18}
    # Wait, the indices in the formula above are 1-based P.
    # Let's map to 0-based:
    # C_00 = P_0 + P_4 - P_8 + P_9
    # C_01 = P_1 + P_5 - P_9 + P_10
    # C_02 = P_2 + P_6 - P_10 + P_11
    # C_10 = P_4 + P_7 - P_12 + P_13
    # C_11 = P_5 + P_8 - P_13 + P_14
    # C_12 = P_6 + P_9 - P_14 + P_15
    # C_20 = P_7 + P_10 - P_15 + P_16
    # C_21 = P_8 + P_11 - P_16 + P_17
    # C_22 = P_9 + P_12 - P_17 + P_18
    
    # Note: The terms P_15..P_22 are the "correction" terms in Laderman.
    # The reconstruction might involve them differently.
    # Actually, Laderman's algorithm is often described with 23 terms where the last 6 are corrections.
    # The reconstruction formula above only uses P_0..P_18?
    # Let's check the number of terms used.
    # If I use this reconstruction, I might be missing the contribution of P_19..P_22.
    # However, if the decomposition is exact, the W matrix will naturally handle it if I define the mapping correctly.
    # But I must define the mapping.
    
    # Let's assume the standard Laderman reconstruction is:
    # C_00 = P_0 + P_4 - P_8 + P_9
    # C_01 = P_1 + P_5 - P_9 + P_10
    # C_02 = P_2 + P_6 - P_10 + P_11
    # C_10 = P_4 + P_7 - P_12 + P_13
    # C_11 = P_5 + P_8 - P_13 + P_14
    # C_12 = P_6 + P_9 - P_14 + P_15
    # C_20 = P_7 + P_10 - P_15 + P_16
    # C_21 = P_8 + P_11 - P_16 + P_17
    # C_22 = P_9 + P_12 - P_17 + P_18
    
    # This uses P_0..P_18. What about P_19..P_22?
    # In some formulations, the terms are different.
    # Let's try to verify numerically.
    
    laderman_recon = [
        # C_00
        {0: 1, 4: 1, 8: -1, 9: 1},
        # C_01
        {1: 1, 5: 1, 9: -1, 10: 1},
        # C_02
        {2: 1, 6: 1, 10: -1, 11: 1},
        # C_10
        {4: 1, 7: 1, 12: -1, 13: 1},
        # C_11
        {5: 1, 8: 1, 13: -1, 14: 1},
        # C_12
        {6: 1, 9: 1, 14: -1, 15: 1},
        # C_20
        {7: 1, 10: 1, 15: -1, 16: 1},
        # C_21
        {8: 1, 11: 1, 16: -1, 17: 1},
        # C_22
        {9: 1, 12: 1, 17: -1, 18: 1},
    ]
    
    rank = 23 * 7
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    for t in range(23):
        x_terms, y_terms = laderman_terms[t]
        
        # Build X_t and Y_t as 2x2 matrices of coefficients
        # X_t[p, q] is a linear combo of A blocks
        # We need to map A blocks to global indices
        # A block (i, j) covers rows 3i..3i+2, cols 3j..3j+2
        # Global index for A[i, j] element (p, q) is (3i+p)*6 + (3j+q)
        
        # For Strassen, we need the 4 elements of X_t and Y_t
        # X_t is 2x2. Let's call its elements x00, x01, x10, x11
        # These are linear combinations of A blocks.
        # Wait, X_t itself is a 2x2 matrix of *blocks*?
        # No, X_t is a 2x2 matrix of *scalars* (coefficients)?
        # No, in Laderman, the "entries" are blocks.
        # So X_t is a 2x2 matrix where each entry is a block of A.
        # So X_t[0,0] is a 2x2 block of A.
        
        # Let's represent X_t[0,0] as a vector of 4 coefficients for the 4 elements of A.
        # Actually, we can just iterate over the 4 elements of the 2x2 Strassen input.
        
        # For each of the 7 Strassen products:
        for s in range(7):
            r = t * 7 + s
            s_x_factors, s_y_factors = strassen_factors[s]
            
            # s_x_factors: list of (p, q, coeff) for X_t[p, q]
            # X_t[p, q] is a linear combination of A blocks.
            # From laderman_terms[t], we have x_terms: list of (i, j, coeff)
            # This defines X_t as a 2x2 matrix of blocks?
            # No, laderman_terms defines the *blocks* that form the 3x3 matrix X.
            # Wait, Laderman operates on 3x3 matrices of blocks.
            # So X is 3x3 blocks.
            # But Strassen operates on 2x2 matrices.
            # So we must map the 3x3 block structure to 2x2?
            # No, Laderman's algorithm is for 3x3 matrix multiplication.
            # The "matrices" in Laderman are 3x3.
            # So X is 3x3.
            # But Strassen is for 2x2.
            # This mismatch means we cannot directly plug Laderman's 3x3 terms into Strassen's 2x2 terms.
            
            # CORRECTION:
            # The strategy is:
            # 1. Partition 6x6 into 3x3 blocks of 2x2.
            # 2. Use Laderman to compute the 3x3 block multiplication.
            #    This requires 23 multiplications of 2x2 blocks.
            # 3. Each 2x2 block multiplication is done via Strassen (7 scalar mults).
            
            # So, for Laderman term t, we have X_t and Y_t which are 2x2 matrices of *blocks*?
            # No, Laderman's terms are linear combinations of the *entries* of the 3x3 matrix.
            # The entries of the 3x3 matrix are 2x2 blocks.
            # So X_t is a linear combination of 2x2 blocks.
            # Thus X_t is itself a 2x2 matrix (of scalars? No, of block elements).
            # Wait, a linear combination of 2x2 matrices is a 2x2 matrix.
            # So X_t is a 2x2 matrix.
            # Yes!
            # So X_t is a 2x2 matrix.
            # And we can apply Strassen to X_t and Y_t.
            
            # So we need to construct the 2x2 matrix X_t.
            # X_t[p, q] = sum_{(i, j, c) in x_terms} c * A_block[i, j][p, q] ?
            # No, x_terms are (i, j, c) where i, j are block indices in 0..2.
            # So X_t = sum c * A_block[i, j].
            # A_block[i, j] is a 2x2 matrix.
            # So X_t is a 2x2 matrix.
            
            # Let's build X_t and Y_t as 2x2 numpy arrays of coefficients?
            # No, we need to map to global U, V vectors.
            
            # For U:
            # U[:, r] should be the coefficients for A that form the first factor of the s-th Strassen product.
            # The s-th Strassen product uses linear combinations of the 4 elements of X_t.
            # s_x_factors gives these combinations.
            # e.g. S1 uses X_t[0,0] + X_t[1,1].
            # X_t[0,0] is the (0,0) element of the matrix X_t.
            # X_t[0,0] = sum_{(i, j, c) in x_terms} c * A_block[i, j][0, 0].
            
            # So, for a given Strassen factor (p, q, coeff_s):
            # We need to add coeff_s * (sum_{(i, j, c) in x_terms} c * A_block[i, j][p, q]) to U.
            
            for p, q, coeff_s in s_x_factors:
                for i, j, coeff_l in x_terms:
                    # A_block[i, j][p, q] is at global index (2i+p)*6 + (2j+q)
                    idx = (2*i + p) * 6 + (2*j + q)
                    U[idx, r] += coeff_l * coeff_s
            
            # Similarly for V
            for p, q, coeff_s in s_y_factors:
                for i, j, coeff_l in y_terms:
                    # B_block[i, j][p, q] is at global index (2i+p)*6 + (2j+q)
                    idx = (2*i + p) * 6 + (2*j + q)
                    V[idx, r] += coeff_l * coeff_s
            
            # Now for W
            # W[:, r] should be the coefficients for C that the s-th Strassen product of t-th Laderman term contributes to.
            # The s-th Strassen product contributes to P_t[ab] with coefficient strassen_recon[ab][s].
            # P_t is a 2x2 matrix.
            # P_t[ab] contributes to C_block[i, j] with coefficient laderman_recon[i*3+j].get(t, 0) ?
            # No, laderman_recon maps C_block indices to P_t indices.
            # We need the reverse: for each P_t, which C_blocks does it contribute to?
            # And with what coefficient?
            # And P_t is a 2x2 matrix. Does it contribute to C_block[i, j] as a whole?
            # Yes, C_block[i, j] is a 2x2 matrix.
            # P_t is a 2x2 matrix.
            # The reconstruction is C_block[i, j] = sum_t coeff * P_t.
            # So P_t[ab] contributes to C_block[i, j][ab].
            
            for ab in range(4): # 00, 01, 10, 11
                p_idx = ab // 2
                q_idx = ab % 2
                coeff_strassen = strassen_recon[ab][s]
                if coeff_strassen == 0:
                    continue
                
                # Find which C_blocks this P_t contributes to
                for c_i in range(3):
                    for c_j in range(3):
                        c_block_idx = c_i * 3 + c_j
                        coeff_laderman = laderman_recon[c_block_idx].get(t, 0)
                        if coeff_laderman == 0:
                            continue
                        
                        # C_block[c_i, c_j] is at rows 2c_i..2c_i+1, cols 2c_j..2c_j+1.
                        # Element [p_idx, q_idx] is at global row 2c_i+p_idx, col 2c_j+q_idx.
                        # C_{row, col} = C_{i, k} where i=row, k=col.
                        # W index is k*n + i = col*6 + row.
                        
                        k_idx = 2*c_j + q_idx
                        i_idx = 2*c_i + p_idx
                        w_idx = k_idx * 6 + i_idx
                        W[w_idx, r] += coeff_strassen * coeff_laderman
    
    # Verify
    U_reshaped = U.reshape(6, 6, -1)
    V_reshaped = V.reshape(6, 6, -1)
    W_reshaped = W.reshape(6, 6, -1)
    reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    target = np.ones((6, 6, 6))
    
    if not np.allclose(reconstructed, target, atol=1e-8):
        # Fallback to schoolbook if Laderman/Strassen fails
        rank = 216
        U = np.zeros((36, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    idx = i*6 + j
                    jk = j*6 + k
                    ki = k*6 + i
                    col = i*36 + j*6 + k
                    U[idx, col] = 1.0
                    V[jk, col] = 1.0
                    W[ki, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
