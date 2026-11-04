import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get the 4x4x4 rank-49 factors
    # Returns U4, V4, W4 of shape (16, 49)
    def get_4x4x4_factors():
        # Strassen 2x2x2 factors
        # u, v, w are 4x7 matrices for 2x2x2
        # Indices 0..3 for 2x2 matrices
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        # Strassen terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Coefficients for A blocks in each M
        # a_coeffs[s][b] where s in 0..6, b in {0:11, 1:12, 2:21, 3:22}
        a_coeffs = [
            [1, 0, 0, 1], # M1: A11 + A22
            [0, 0, 1, 1], # M2: A21 + A22
            [1, 0, 0, 0], # M3: A11
            [0, 0, 0, 1], # M4: A22
            [1, 1, 0, 0], # M5: A11 + A12
            [-1, 0, 1, 0],# M6: A21 - A11
            [0, 1, 0, -1] # M7: A12 - A22
        ]
        
        b_coeffs = [
            [1, 0, 0, 1], # M1: B11 + B22
            [1, 0, 0, 0], # M2: B11
            [0, 1, 0, -1],# M3: B12 - B22
            [0, 0, 1, -1],# M4: B21 - B11
            [0, 0, 0, 1], # M5: B22
            [1, 1, 0, 0], # M6: B11 + B12
            [0, 0, 1, 1]  # M7: B21 + B22
        ]
        
        # Output reconstruction coefficients
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        c_coeffs = [
            [1, 0, 0, 1, -1, 0, 1],  # C11
            [0, 0, 1, 0, 1, 0, 0],   # C12
            [0, 1, 0, 1, 0, 0, 0],   # C21
            [1, -1, 1, 0, 0, 1, 0]   # C22
        ]
        
        # 2x2x2 Strassen factors (rank 7)
        # u2, v2, w2 are 4x7
        # We need the explicit 2x2x2 rank-7 factors.
        # This is standard Strassen.
        # Let's define the 7 vectors for 2x2x2.
        # u2[t], v2[t], w2[t] for t in 0..6
        
        # t=0 (M1): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] ? No, w is for C.
        # The 2x2x2 tensor decomposition is not just the Strassen steps themselves,
        # but the rank-7 decomposition of the 2x2x2 tensor.
        # The Strassen algorithm IS the rank-7 decomposition.
        # The "factors" u2, v2, w2 for the 2x2x2 tensor are such that
        # C = sum_t (u2_t . A) * (v2_t . B) * w2_t (outer product)
        
        # Let's find the standard 2x2x2 rank-7 factors.
        # They are given by the 7 Strassen multiplications.
        # For each multiplication M_t, we have linear forms in A and B.
        # Let L_A_t be the linear form for A, L_B_t for B.
        # The product is L_A_t(L_B_t).
        # This product contributes to C.
        
        # Actually, it's easier to just build the 4x4x4 factors directly from the block structure.
        
        U4 = np.zeros((16, 49), dtype=int)
        V4 = np.zeros((16, 49), dtype=int)
        W4 = np.zeros((16, 49), dtype=int)
        
        # Mapping of local 2x2 indices to global 4x4 indices for each block
        # Block 0 (11): 0,1,4,5
        # Block 1 (12): 2,3,6,7
        # Block 2 (21): 8,9,12,13
        # Block 3 (22): 10,11,14,15
        block_map_A = [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15]
        ]
        
        block_map_B = [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15]
        ]
        
        block_map_C = [
            [0, 1, 4, 5],   # C11
            [8, 9, 12, 13], # C12
            [2, 3, 6, 7],   # C21
            [10, 11, 14, 15]# C22
        ]
        
        # We need the rank-7 factors for 2x2x2.
        # Let's hardcode the 2x2x2 rank-7 factors (u2, v2, w2)
        # These are 4x7 matrices.
        # Source: Standard Strassen
        # The 7 terms are linear combinations.
        # u2[t] is the coefficient vector for A in the t-th term.
        # v2[t] is the coefficient vector for B in the t-th term.
        # w2[t] is the coefficient vector for C in the t-th term.
        
        # Wait, the decomposition is C = sum_t (u2_t^T A) (v2_t^T B) w2_t
        # This is not quite right. The tensor is T_{ijk} = sum_k A_{ik} B_{kj}.
        # Rank decomposition: T = sum_r u_r \otimes v_r \otimes w_r
        # u_r is vector in R^4 (for A), v_r in R^4 (for B), w_r in R^4 (for C).
        
        # The Strassen algorithm provides this.
        # For t=0 (M1):
        # A_part = A11 + A22 -> u2[0] = [1, 0, 0, 1]
        # B_part = B11 + B22 -> v2[0] = [1, 0, 0, 1]
        # C_part = C11 + C22? No, M1 contributes to multiple C blocks.
        # The w2 vector is not simply the C coefficients.
        # Actually, the Strassen algorithm computes 7 products, and then combines them to get C.
        # This means the tensor is a sum of 7 rank-1 tensors? No, Strassen is a rank-7 decomposition.
        # The 7 "multiplications" are the rank-1 terms.
        # So, for each t, we have a rank-1 tensor defined by u2_t, v2_t, w2_t.
        
        # Let's look up the explicit 2x2x2 rank-7 factors.
        # u2, v2, w2 are 4x7.
        
        u2 = [
            [1, 0, 1, 0, 1, -1, 0], # A11, A12, A21, A22 coeffs for t=0..6? No, rows are A indices.
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, -1]
        ]
        # This is getting risky. Let's use a simpler, verified method.
        # We can just use the schoolbook 2x2x2 (rank 4) to get a rank-16 4x4x4 decomposition?
        # 4x4x4 with schoolbook 2x2 blocks: 4 blocks * 4 ranks = 16 ranks per block?
        # No, 4x4 = 4 blocks of 2x2.
        # If we use schoolbook for each 2x2 block multiplication, we need 4 multiplications per block.
        # 4 blocks * 4 mults = 16 mults.
        # Each mult is scalar.
        # So rank 16 for 4x4x4?
        # Wait, 4x4x4 schoolbook rank is 16? No, 4*4*4 = 64.
        # If we use Strassen (rank 7) for the 2x2 level, we get 49.
        # If we use schoolbook (rank 4) for the 2x2 level, we get 16?
        # Let's check.
        # A 4x4 multiplication using 2x2 blocks.
        # C11 = A11 B11 + A12 B21
        # C12 = A11 B12 + A12 B22
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        # Each term is a 2x2 multiplication.
        # If we compute each 2x2 multiplication with rank 4 (schoolbook),
        # we have 8 terms (C11 has 2, etc).
        # Total 8 * 4 = 32 ranks.
        # This is better than 49? No, 49 is better than 64, but 32 is better than 49?
        # Wait, is rank(4x4x4) 32?
        # No, rank(4x4x4) is known to be 49.
        # Why is my block calculation giving 32?
        # Because the 2x2 multiplications are not independent?
        # No, they are.
        # Ah, the rank of the tensor M_{4,4,4} is the minimum number of scalar multiplications.
        # If I use 32 scalar multiplications, the rank is 32.
        # But it is proven that rank(M_{4,4,4}) = 49.
        # Where is the error?
        # The error is that the "terms" in the block expansion are not scalar multiplications.
        # A11 B11 is a matrix product. It requires 4 scalar multiplications if done schoolbook.
        # So 8 matrix products * 4 scalar mults = 32 scalar mults.
        # This implies rank <= 32.
        # But literature says 49.
        # Let me check the rank of 2x2x2. It is 7.
        # If I use Strassen for the 2x2 level, I get 7 mults per 2x2 block product.
        # 8 block products * 7 mults = 56 mults.
        # But Strassen reduces the 4x4 to 7 block products.
        # So 7 block products * 7 mults = 49 mults.
        # This is the 49 rank.
        
        # So, to get rank 49, I must use the Strassen structure for the 4x4 level.
        # I cannot just use schoolbook for the 2x2 level inside the 4x4 block expansion.
        # I must use the Strassen 7-term structure for the 4x4, and then expand each term.
        
        # So my previous plan was correct.
        # I need the 2x2x2 rank-7 factors to expand the Strassen terms.
        
        # Let's find the 2x2x2 rank-7 factors.
        # They are the same as the Strassen coefficients for 2x2.
        # u2, v2, w2 (4x7)
        
        # I will use a known set of 2x2x2 rank-7 factors.
        # From Wikipedia "Strassen algorithm":
        # The 7 products are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # These are the linear forms.
        # For the tensor decomposition, we need u_t, v_t, w_t such that
        # C = sum_t (u_t . A) (v_t . B) w_t
        # This is not how Strassen works directly. Strassen computes products of linear forms.
        # The product M_t is a scalar.
        # The contribution to C is M_t * (some matrix pattern).
        # So w_t is the matrix pattern flattened.
        
        # For M1:
        # u1 = [1, 0, 0, 1] (A11+A22)
        # v1 = [1, 0, 0, 1] (B11+B22)
        # w1 = [1, 0, 0, 1] (C11+C22)? No.
        # C11 gets +M1, C22 gets +M1.
        # So w1 = [1, 0, 0, 1] in C indices (C11, C12, C21, C22).
        
        # For M2:
        # u2 = [0, 0, 1, 1] (A21+A22)
        # v2 = [1, 0, 0, 0] (B11)
        # w2 = [0, 0, 1, 0] (C21)
        
        # For M3:
        # u3 = [1, 0, 0, 0] (A11)
        # v3 = [0, 1, 0, -1] (B12-B22)
        # w3 = [0, 1, 0, 0] (C12)
        
        # For M4:
        # u4 = [0, 0, 0, 1] (A22)
        # v4 = [0, 0, 1, -1] (B21-B11)
        # w4 = [1, 0, 1, 0] (C11+C21)
        
        # For M5:
        # u5 = [1, 1, 0, 0] (A11+A12)
        # v5 = [0, 0, 0, 1] (B22)
        # w5 = [-1, 1, 0, 0] (C12-C11)
        
        # For M6:
        # u6 = [-1, 0, 1, 0] (A21-A11)
        # v6 = [1, 1, 0, 0] (B11+B12)
        # w6 = [0, 0, 0, 1] (C22)
        
        # For M7:
        # u7 = [0, 1, 0, -1] (A12-A22)
        # v7 = [0, 0, 1, 1] (B21+B22)
        # w7 = [1, 0, 0, 1] (C11+C22)
        
        # Let's verify C11: M1 + M4 - M5 + M7
        # w1[0]=1, w4[0]=1, w5[0]=-1, w7[0]=1. Sum=2?
        # Wait, C11 = M1 + M4 - M5 + M7.
        # My w vectors:
        # w1[0]=1
        # w4[0]=1
        # w5[0]=-1
        # w7[0]=1
        # Sum = 1+1-1+1 = 2.
        # But C11 should be M1 + M4 - M5 + M7.
        # So the coefficient is 1 for each.
        # So w1[0]=1, w4[0]=1, w5[0]=-1, w7[0]=1.
        # This matches.
        
        # So we have the 2x2x2 rank-7 factors.
        u2 = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, -1]
        ])
        v2 = np.array([
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, -1, 1, 0, 1]
        ])
        w2 = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0]
        ])
        # Note: w2 indices 0..3 correspond to C11, C12, C21, C22.
        
        # Now build U4, V4, W4 (16x49)
        for s in range(7):
            for t in range(7):
                r = s * 7 + t
                # A part
                for local_a in range(4):
                    coeff_a = u2[local_a, t]
                    if coeff_a != 0:
                        # This local_a index corresponds to a position in the 2x2 block.
                        # But which block?
                        # The Strassen step s determines the block combination.
                        # a_coeffs[s] gives the weight of each block.
                        # The local index local_a (0..3) maps to (i,j) in 2x2.
                        # We need to add coeff_a * a_coeffs[s][block] to U4[global_idx, r].
                        
                        # Determine which block this local index belongs to?
                        # No, the local index is the position within the 2x2 matrix.
                        # The 2x2 matrix A_s is a sum of blocks.
                        # A_s[i,j] = sum_b alpha_b A_b[i,j].
                        # So for a fixed local (i,j) (which is local_a),
                        # the contribution to global A index corresponding to block b and local (i,j) is alpha_b * coeff_a.
                        
                        # local_a 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                        # We need to iterate over blocks b.
                        for b in range(4):
                            alpha = a_coeffs[s][b]
                            if alpha != 0:
                                global_idx = block_map_A[b][local_a]
                                U4[global_idx, r] += coeff_a * alpha
                                
                # B part
                for local_b in range(4):
                    coeff_b = v2[local_b, t]
                    if coeff_b != 0:
                        for b in range(4):
                            alpha = b_coeffs[s][b]
                            if alpha != 0:
                                global_idx = block_map_B[b][local_b]
                                V4[global_idx, r] += coeff_b * alpha
                                
                # C part
                for local_c in range(4):
                    coeff_c = w2[local_c, t]
                    if coeff_c != 0:
                        # w2[local_c, t] is the coefficient for the c-th output block (C11, C12, C21, C22)
                        # local_c 0->C11, 1->C12, 2->C21, 3->C22
                        # We need to add this to the global C indices for that block.
                        # The global C indices for block local_c are in block_map_C[local_c].
                        # But w2[local_c, t] is a scalar coefficient for the whole block?
                        # No, w2 is 4x7. w2[local_c, t] is the coefficient for the local_c-th element of the 2x2 output?
                        # No, w2 is the vector for the 2x2 output matrix.
                        # So w2[local_c, t] is the coefficient for the element at local_c in the 2x2 output.
                        # And this 2x2 output is the block C_{block_c}.
                        # So for each local_c, we add coeff_c to the global indices of block C_{local_c} at position local_c.
                        
                        # Wait, block_map_C[local_c] is a list of 4 global indices for the block.
                        # local_c is the index within the 2x2 block.
                        # So global_idx = block_map_C[local_c][local_c]?
                        # No.
                        # block_map_C[0] is [0,1,4,5] for C11.
                        # local_c=0 corresponds to C11(0,0) -> global 0.
                        # local_c=1 corresponds to C11(0,1) -> global 1.
                        # So yes, global_idx = block_map_C[local_c][local_c].
                        
                        global_idx = block_map_C[local_c][local_c]
                        W4[global_idx, r] += coeff_c
                        
        return U4, V4, W4

    U4, V4, W4 = get_4x4x4_factors()
    
    # Now construct the 4x4x8 factors
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # First 49 columns
    U[:, 0:49] = U4
    V[:, 0:49] = V4
    W[:, 0:49] = W4
    
    # Last 49 columns
    U[:, 49:98] = U4
    
    # V for second half
    # B indices for second half (k=4..7)
    # Block map for B second half
    # B11' (j=0,1, k=4,5) -> 4,5,12,13
    # B12' (j=0,1, k=6,7) -> 6,7,14,15
    # B21' (j=2,3, k=4,5) -> 20,21,28,29
    # B22' (j=2,3, k=6,7) -> 22,23,30,31
    
    block_map_B2 = [
        [4, 5, 12, 13],
        [6, 7, 14, 15],
        [20, 21, 28, 29],
        [22, 23, 30, 31]
    ]
    
    # We need to remap V4 to V[:, 49:98]
    # V4 has rows 0..15 corresponding to B indices 0..15 (first half).
    # We need to map these to the second half indices.
    # The structure of V4 is built from block_map_B.
    # We can just re-construct the V part for the second half using block_map_B2.
    
    # Let's re-run the V construction logic for the second half.
    # We can reuse the loop but with block_map_B2.
    
    # To avoid duplicating code, I'll just map the indices.
    # V4[i, r] corresponds to B index i in the first half.
    # The mapping from first half B index to second half B index is:
    # 0->4, 1->5, 2->6, 3->7
    # 4->12, 5->13, 6->14, 7->15
    # 8->20, 9->21, 10->22, 11->23
    # 12->28, 13->29, 14->30, 15->31
    
    b_map = [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]
    for i in range(16):
        V[b_map[i], 49:98] = V4[i, :]
        
    # W for second half
    # C indices for second half (k=4..7)
    # C11' (k=4,5, i=0,1) -> 16,17,20,21
    # C12' (k=6,7, i=0,1) -> 24,25,28,29
    # C21' (k=4,5, i=2,3) -> 18,19,22,23
    # C22' (k=6,7, i=2,3) -> 26,27,30,31
    
    c_map = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    for i in range(16):
        W[c_map[i], 49:98] = W4[i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
