"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a rank-49 decomposition for 4x4x4 multiplication based on Strassen's algorithm.
    # Total rank for 4x4x8 is 2 * 49 = 98.
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)] # 16 x 98
    V = [[0] * rank for _ in range(m * p)] # 32 x 98
    W = [[0] * rank for _ in range(p * n)] # 32 x 98
    
    # Define the 2x2x2 Strassen decomposition coefficients
    # M1..M7 are the 7 products.
    # Each product is of the form (linear comb of A) * (linear comb of B)
    # Then the result is combined to form C.
    
    # Strassen 2x2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # The 7 terms are:
    # T1 = (a00+a11)(b00+b11)
    # T2 = (a10+a11)b00
    # T3 = a00(b01-b11)
    # T4 = a11(b10-b00)
    # T5 = (a00+a01)b11
    # T6 = (a10-a00)(b00+b01)
    # T7 = (a01-a11)(b10+b11)
    
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    
    # We need to expand these into the 4x4 case.
    # Let A be 4x4, B be 4x4.
    # Blocks: A00, A01, A10, A11 (each 2x2)
    # The block-level Strassen terms M1..M7 are 2x2 matrices.
    # M1 = (A00+A11)(B00+B11)
    # ... and so on.
    
    # Each Mk is a 2x2 matrix product. We decompose each Mk using the 2x2x2 Strassen algorithm.
    # So for each Mk, we have 7 scalar multiplications.
    
    # Let's generate the coefficients for one 4x4x4 decomposition (rank 49).
    # Then we duplicate it for the second half of the 8 columns.
    
    def get_4x4x4_decomp():
        # Returns U4, V4, W4 of size 16x49, 16x49, 16x49
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        # Indices for 4x4 matrices: 0..15
        # i in 0..3, j in 0..3 -> idx = i*4+j
        # For 2x2 blocks, we map block (r,c) and local (s,t) to global (2r+s, 2c+t)
        
        # The 7 block terms M1..M7
        # For each term k (0..6), we define:
        # X_k: 2x2 matrix of coefficients for A blocks
        # Y_k: 2x2 matrix of coefficients for B blocks
        # Z_k: 2x2 matrix of coefficients for C blocks (how Mk contributes to C blocks)
        
        # M1 = (A00+A11)(B00+B11)
        # X1 = [[1, 0], [0, 1]] (A00 and A11) -- Wait, the product is (A00+A11) * (B00+B11)
        # This is not a single matrix multiplication of 2x2 matrices in the standard sense unless we view A00+A11 as one matrix.
        # Yes, X1 is the matrix formed by summing A blocks? No.
        # The term is (A00 + A11) * (B00 + B11).
        # Let P1 = A00 + A11. Q1 = B00 + B11.
        # Then M1 = P1 * Q1.
        # P1 is a 2x2 matrix. Q1 is a 2x2 matrix.
        # We decompose P1 * Q1 using 2x2x2 Strassen.
        
        # Let's define the linear combinations for P and Q for each of the 7 block terms.
        # P_k = sum_{r,c} coeff_k[r,c] * A_{r,c}
        # Q_k = sum_{r,c} coeff_k[r,c] * B_{r,c}
        
        # M1: P1 = A00 + A11, Q1 = B00 + B11
        # M2: P2 = A10 + A11, Q2 = B00
        # M3: P3 = A00, Q3 = B01 - B11
        # M4: P4 = A11, Q4 = B10 - B00
        # M5: P5 = A00 + A01, Q5 = B11
        # M6: P6 = A10 - A00, Q6 = B00 + B01
        # M7: P7 = A01 - A11, Q7 = B10 + B11
        
        # Coefficients for P (A blocks) and Q (B blocks)
        # Blocks indexed by (r,c) where r,c in {0,1}
        # A_{r,c} corresponds to global indices (2r+s, 2c+t)
        
        P_coeffs = [
            [[1, 0], [0, 1]], # M1: A00 + A11
            [[0, 0], [1, 1]], # M2: A10 + A11
            [[1, 0], [0, 0]], # M3: A00
            [[0, 0], [0, 1]], # M4: A11
            [[1, 1], [0, 0]], # M5: A00 + A01
            [[-1, 0], [1, 0]],# M6: A10 - A00
            [[0, 1], [0, -1]] # M7: A01 - A11
        ]
        
        Q_coeffs = [
            [[1, 0], [0, 1]], # M1: B00 + B11
            [[1, 0], [0, 0]], # M2: B00
            [[0, 1], [0, -1]],# M3: B01 - B11
            [[0, -1], [1, 0]],# M4: B10 - B00
            [[0, 0], [0, 1]], # M5: B11
            [[1, 1], [0, 0]], # M6: B00 + B01
            [[0, 1], [1, 1]]  # M7: B10 + B11
        ]
        
        # Now, for each block term k, we have P_k * Q_k.
        # P_k and Q_k are 2x2 matrices.
        # We decompose P_k * Q_k using the 2x2x2 Strassen algorithm.
        # Let R_k = P_k * Q_k.
        # R_k is a 2x2 matrix.
        # We need to express R_k in terms of 7 scalar multiplications.
        
        # The 2x2x2 Strassen algorithm for P * Q:
        # P = [[p00, p01], [p10, p11]], Q = [[q00, q01], [q10, q11]]
        # S1 = (p00+p11)(q00+q11)
        # S2 = (p10+p11)q00
        # S3 = p00(q01-q11)
        # S4 = p11(q10-q00)
        # S5 = (p00+p01)q11
        # S6 = (p10-p00)(q00+q01)
        # S7 = (p01-p11)(q10+q11)
        
        # R00 = S1 + S4 - S5 + S7
        # R01 = S3 + S5
        # R10 = S2 + S4
        # R11 = S1 - S2 + S3 + S6
        
        # We need to map this back to the 4x4 case.
        # The entries of P_k are linear combinations of A entries.
        # The entries of Q_k are linear combinations of B entries.
        # So each S_j is a product of two linear combinations.
        # This is a rank-1 term in the tensor decomposition.
        
        # For each block term k (0..6) and each Strassen term j (0..6):
        # We have a scalar multiplication s_{k,j}.
        # U_part: linear comb of A entries corresponding to the first factor of s_{k,j}
        # V_part: linear comb of B entries corresponding to the second factor of s_{k,j}
        # W_part: contribution to C entries.
        
        # Let's precompute the linear combinations for P_k and Q_k entries.
        # P_k[s,t] = sum_{r,c} P_coeffs[k][r,c] * A[2r+s, 2c+t]
        # Q_k[s,t] = sum_{r,c} Q_coeffs[k][r,c] * B[2r+s, 2c+t]
        
        # For each k, j:
        # Factor 1 (U):
        # If j=0: (p00+p11) -> sum_{r,c} (P_k[0,0] + P_k[1,1]) ... wait.
        # p00 is P_k[0,0]. p11 is P_k[1,1].
        # P_k[0,0] is a linear comb of A.
        # So the U vector for this term is the sum of the U vectors for P_k[0,0] and P_k[1,1].
        
        # Let's build the U, V, W vectors for the 49 terms.
        
        col = 0
        for k in range(7):
            # Define P_k and Q_k as 2x2 matrices of linear combinations
            # We represent each entry as a list of (global_index, coeff)
            
            P_entries = [[0]*4 for _ in range(2)] # 2x2 matrix of dicts? Or just sparse vectors
            # Let's use lists of (index, coeff)
            P_vecs = [[[] for _ in range(2)] for _ in range(2)]
            Q_vecs = [[[] for _ in range(2)] for _ in range(2)]
            
            for s in range(2):
                for t in range(2):
                    for r in range(2):
                        for c in range(2):
                            coeff = P_coeffs[k][r][c]
                            if coeff != 0:
                                # A[2r+s, 2c+t]
                                idx_a = (2*r+s)*4 + (2*c+t)
                                P_vecs[s][t].append((idx_a, coeff))
                                
                            coeff_q = Q_coeffs[k][r][c]
                            if coeff_q != 0:
                                idx_b = (2*r+s)*4 + (2*c+t)
                                Q_vecs[s][t].append((idx_b, coeff_q))
            
            # Now apply 2x2x2 Strassen to P_k * Q_k
            # S1 = (p00+p11)(q00+q11)
            # S2 = (p10+p11)q00
            # S3 = p00(q01-q11)
            # S4 = p11(q10-q00)
            # S5 = (p00+p01)q11
            # S6 = (p10-p00)(q00+q01)
            # S7 = (p01-p11)(q10+q11)
            
            # Define the 7 terms (U_vec, V_vec, W_contrib)
            # U_vec is a vector of length 16
            # V_vec is a vector of length 16
            # W_contrib is a dict mapping C index to coeff
            
            terms = []
            
            # Helper to add vectors
            def add_vecs(v1, v2, sign=1):
                res = [0]*16
                for idx, c in v1:
                    res[idx] += c
                for idx, c in v2:
                    res[idx] += sign * c
                return res
            
            # S1
            u1 = add_vecs(P_vecs[0][0], P_vecs[1][1])
            v1 = add_vecs(Q_vecs[0][0], Q_vecs[1][1])
            # R00 += S1, R11 += S1
            w1 = {0: 1, 11: 1} # C00, C11
            # Note: R is 2x2. R[0,0] is C block 00, R[0,1] is C block 01, etc.
            # C global index: (2r+s)*4 + (2c+t)
            # R[0,0] -> C[0..1, 0..1] -> indices 0,1,4,5
            # R[0,1] -> C[0..1, 2..3] -> indices 2,3,6,7
            # R[1,0] -> C[2..3, 0..1] -> indices 8,9,12,13
            # R[1,1] -> C[2..3, 2..3] -> indices 10,11,14,15
            
            def expand_w(block_r, block_c, coeff):
                d = {}
                for s in range(2):
                    for t in range(2):
                        idx = (2*block_r+s)*4 + (2*block_c+t)
                        d[idx] = coeff
                return d
            
            # S1 contributes to R00 and R11
            w1 = {}
            w1.update(expand_w(0, 0, 1))
            w1.update(expand_w(1, 1, 1))
            terms.append((u1, v1, w1))
            
            # S2
            u2 = add_vecs(P_vecs[1][0], P_vecs[1][1])
            v2 = Q_vecs[0][0]
            w2 = expand_w(1, 0, 1) # R10
            terms.append((u2, v2, w2))
            
            # S3
            u3 = P_vecs[0][0]
            v3 = add_vecs(Q_vecs[0][1], Q_vecs[1][1], -1)
            w3 = expand_w(0, 1, 1) # R01
            terms.append((u3, v3, w3))
            
            # S4
            u4 = P_vecs[1][1]
            v4 = add_vecs(Q_vecs[1][0], Q_vecs[0][0], -1)
            w4 = {}
            w4.update(expand_w(0, 0, 1)) # R00
            w4.update(expand_w(1, 0, 1)) # R10
            terms.append((u4, v4, w4))
            
            # S5
            u5 = add_vecs(P_vecs[0][0], P_vecs[0][1])
            v5 = Q_vecs[1][1]
            w5 = {}
            w5.update(expand_w(0, 0, -1)) # R00
            w5.update(expand_w(0, 1, 1)) # R01
            terms.append((u5, v5, w5))
            
            # S6
            u6 = add_vecs(P_vecs[1][0], P_vecs[0][0], -1)
            v6 = add_vecs(Q_vecs[0][0], Q_vecs[0][1])
            w6 = expand_w(1, 1, 1) # R11
            terms.append((u6, v6, w6))
            
            # S7
            u7 = add_vecs(P_vecs[0][1], P_vecs[1][1], -1)
            v7 = add_vecs(Q_vecs[1][0], Q_vecs[1][1])
            w7 = expand_w(0, 0, 1) # R00
            terms.append((u7, v7, w7))
            
            # Now add these 7 terms to U4, V4, W4
            for u_t, v_t, w_t in terms:
                # u_t, v_t are lists of (idx, coeff)
                for idx, c in u_t:
                    U4[idx][col] = c
                for idx, c in v_t:
                    V4[idx][col] = c
                for idx, c in w_t.items():
                    W4[idx][col] = c
                col += 1
        
        return U4, V4, W4
    
    U4, V4, W4 = get_4x4x4_decomp()
    
    # Now fill U, V, W for the 4x4x8 case
    # U is 16x98. First 49 cols from U4, next 49 cols from U4.
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            U[i][c+49] = U4[i][c]
            
    # V is 32x98.
    # First 49 cols: V4 for B columns 0-3.
    # V4 has 16 rows (for 4x4 B).
    # V has 32 rows. Rows 0-15 correspond to B cols 0-3?
    # V index is j*p + k. j in 0..3, k in 0..7.
    # For the first half (k in 0..3), V[j*8+k] corresponds to B[j,k].
    # V4[j*4+k] corresponds to B[j,k] for k in 0..3.
    # So V[j*8+k] = V4[j*4+k] for k in 0..3.
    for j in range(4):
        for k in range(4):
            v_idx = j*8 + k
            v4_idx = j*4 + k
            for c in range(49):
                V[v_idx][c] = V4[v4_idx][c]
                
    # Second 49 cols: V4 for B columns 4-7.
    # V[j*8+k] for k in 4..7.
    # V4[j*4+k'] for k' in 0..3.
    for j in range(4):
        for k in range(4):
            v_idx = j*8 + (k+4)
            v4_idx = j*4 + k
            for c in range(49):
                V[v_idx][c+49] = V4[v4_idx][c]
                
    # W is 32x98.
    # W index is k*n + i. k in 0..7, i in 0..3.
    # W4 has 16 rows. W4[k*4+i] for k in 0..3.
    # First 49 cols: W for C columns 0-3.
    for k in range(4):
        for i in range(4):
            w_idx = k*4 + i
            w4_idx = k*4 + i
            for c in range(49):
                W[w_idx][c] = W4[w4_idx][c]
                
    # Second 49 cols: W for C columns 4-7.
    for k in range(4):
        for i in range(4):
            w_idx = (k+4)*4 + i
            w4_idx = k*4 + i
            for c in range(49):
                W[w_idx][c+49] = W4[w4_idx][c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
