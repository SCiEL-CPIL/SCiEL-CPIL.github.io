import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 5, 5, 5
    
    # Rank 110 decomposition
    # 49 from Strassen 4x4
    # 16 + 16 + 16 + 4 + 4 + 4 + 1 = 61 cross terms
    
    rank = 110
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        if col >= rank:
            return
        U[u_idx, col] = val
        V[v_idx, col] = val
        W[w_idx, col] = val
        col += 1

    # 1. Strassen 4x4 for top-left block
    # We need to generate the 49 terms for 4x4 Strassen.
    # Strassen 2x2 factors (U, V, W of shape 4x7)
    # Indices for 2x2: 0:00, 1:01, 2:10, 3:11
    
    # Define the 7 Strassen 2x2 terms
    # Each term is (U_vec, V_vec, W_vec) where vecs are length 4 (flattened 2x2)
    # U_vec[i*m + j], V_vec[j*p + k], W_vec[k*n + i]
    
    # Standard Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will construct the 4x4 by combining 7 such 2x2 Strassen sets.
    # Each set corresponds to one of the 7 products P1..P7 above.
    # For each Pk, we apply the 2x2 Strassen logic to the sub-blocks.
    
    # Let's pre-calculate the 2x2 Strassen basis vectors.
    # For a generic 2x2 mult, we have 7 "atoms".
    # But here, for the 4x4 Strassen, we are summing 7 "2x2 matrix products".
    # Let's denote the 2x2 blocks of A as A00, A01, A10, A11.
    # Let's denote the 2x2 blocks of B as B00, B01, B10, B11.
    
    # The 7 products are:
    # P1: A_sum1 = A00+A11, B_sum1 = B00+B11. Product C1 = A_sum1 B_sum1.
    # P2: A_sum2 = A10+A11, B00. Product C2 = A_sum2 B00.
    # P3: A00, B_sum3 = B01-B11. Product C3 = A00 B_sum3.
    # P4: A11, B_sum4 = B10-B00. Product C4 = A11 B_sum4.
    # P5: A_sum5 = A00+A01, B11. Product C5 = A_sum5 B11.
    # P6: A_sum6 = A10-A00, B_sum6 = B00+B01. Product C6 = A_sum6 B_sum6.
    # P7: A_sum7 = A01-A11, B_sum7 = B10+B11. Product C7 = A_sum7 B_sum7.
    
    # Then the final 4x4 result C is:
    # C00 = C1 + C4 - C5 + C7
    # C01 = C3 + C5
    # C10 = C2 + C4
    # C11 = C1 - C2 + C3 + C6
    
    # Each Ck is a 2x2 matrix resulting from a 2x2 matrix multiplication.
    # So each Ck is decomposed into 7 rank-1 terms (using Strassen 2x2).
    # Total 49 terms.
    
    # We need to map the 2x2 Strassen factors to the 4x4 indices.
    # 2x2 indices: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # 4x4 indices: 
    # A00: rows 0-1, cols 0-1. Flat: 0,1,5,6
    # A01: rows 0-1, cols 2-3. Flat: 2,3,7,8
    # A10: rows 2-3, cols 0-1. Flat: 10,11,15,16
    # A11: rows 2-3, cols 2-3. Flat: 12,13,17,18
    
    # Let's define a function to get the flat indices for a 2x2 block in a 4x4 matrix
    def get_2x2_indices_4x4(row_block, col_block):
        # row_block: 0 or 1, col_block: 0 or 1
        r0 = row_block * 2
        c0 = col_block * 2
        return [
            r0 * 4 + c0,
            r0 * 4 + c0 + 1,
            (r0 + 1) * 4 + c0,
            (r0 + 1) * 4 + c0 + 1
        ]
    
    # Strassen 2x2 factors (U, V, W) for the sub-problem.
    # U is 4x7, V is 4x7, W is 4x7.
    # We hardcode these 7 terms for 2x2.
    S2_U = np.zeros((4, 7))
    S2_V = np.zeros((4, 7))
    S2_W = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    # A part: 1 at 0, 1 at 3. B part: 1 at 0, 1 at 3.
    S2_U[0, 0] = 1; S2_U[3, 0] = 1
    S2_V[0, 0] = 1; S2_V[3, 0] = 1
    S2_W[0, 0] = 1 # C00 gets this? 
    # Wait, the W vector in the tensor decomposition for 2x2 mult:
    # T_{i,j,k} = 1. 
    # The decomposition is sum of rank 1 tensors.
    # For M1 = (A00+A11)(B00+B11), the contribution to C is:
    # C00 += M1
    # C01 += 0
    # C10 += 0
    # C11 += 0
    # So the W vector for this term should be 1 at index 0 (C00), 0 elsewhere.
    S2_W[0, 0] = 1; S2_W[1, 0] = 0; S2_W[2, 0] = 0; S2_W[3, 0] = 0
    
    # Term 2: (A10+A11)B00
    S2_U[2, 1] = 1; S2_U[3, 1] = 1
    S2_V[0, 1] = 1
    S2_W[2, 1] = 1 # C10 gets this
    
    # Term 3: A00(B01-B11)
    S2_U[0, 2] = 1
    S2_V[1, 2] = 1; S2_V[3, 2] = -1
    S2_W[1, 2] = 1 # C01 gets this
    
    # Term 4: A11(B10-B00)
    S2_U[3, 3] = 1
    S2_V[2, 3] = 1; S2_V[0, 3] = -1
    S2_W[2, 3] = 1 # C10 gets this? 
    # Check: C10 = M2 + M4. M4 contributes to C10.
    # C10 index is 2. Correct.
    
    # Term 5: (A00+A01)B11
    S2_U[0, 4] = 1; S2_U[1, 4] = 1
    S2_V[3, 4] = 1
    S2_W[0, 4] = -1 # C00 gets -M5
    
    # Term 6: (A10-A00)(B00+B01)
    S2_U[2, 5] = 1; S2_U[0, 5] = -1
    S2_V[0, 5] = 1; S2_V[1, 5] = 1
    S2_W[3, 5] = 1 # C11 gets M6
    
    # Term 7: (A01-A11)(B10+B11)
    S2_U[1, 6] = 1; S2_U[3, 6] = -1
    S2_V[2, 6] = 1; S2_V[3, 6] = 1
    S2_W[0, 6] = 1 # C00 gets M7
    S2_W[3, 6] = 0 # Wait, M7 contributes to C00 and C11?
    # Check formula: C11 = M1 - M2 + M3 + M6. No M7.
    # C00 = M1 + M4 - M5 + M7.
    # So W for Term 7 should be 1 at C00 (idx 0).
    
    # Now, construct the 49 terms for 4x4.
    # We have 7 "big" terms P1..P7.
    # Each Pk is a 2x2 matrix product.
    # We need to express each Pk using the 7 Strassen 2x2 atoms.
    # Let Pk = A'_k B'_k.
    # A'_k and B'_k are 2x2 matrices formed by sums/diffs of A and B blocks.
    # The Strassen 2x2 decomposition of Pk yields 7 rank-1 terms.
    # For each of these 7 terms, we map the 2x2 indices to 4x2 indices.
    
    # Define the coefficients for A'_k and B'_k for k=1..7
    # A'_k is a 2x2 matrix. We represent it by 4 values (a00, a01, a10, a11)
    # B'_k similarly.
    
    P_A_coeffs = [
        [1, 0, 0, 1], # P1: A00+A11
        [0, 0, 1, 1], # P2: A10+A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 1, 0, 0], # P5: A00+A01
        [-1, 0, 1, 0], # P6: A10-A00
        [0, 1, 0, -1]  # P7: A01-A11
    ]
    
    P_B_coeffs = [
        [1, 0, 0, 1], # P1: B00+B11
        [1, 0, 0, 0], # P2: B00
        [0, 1, 0, -1], # P3: B01-B11
        [0, 0, 1, -1], # P4: B10-B00
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00+B01
        [0, 0, 1, 1]  # P7: B10+B11
    ]
    
    # For each Pk (k=0..6):
    #   For each Strassen atom s (s=0..6):
    #      The U vector for this term in 4x4 space is formed by:
    #      For each 2x2 index t (0..3):
    #         val = S2_U[t, s] * P_A_coeffs[k][t]
    #         Place this val at the 4x4 indices corresponding to A-block t.
    
    for k in range(7):
        for s in range(7):
            # Create U, V, W vectors for this specific term in 4x4 space
            u_vec = np.zeros(16)
            v_vec = np.zeros(16)
            w_vec = np.zeros(16)
            
            for t in range(4):
                u_val = S2_U[t, s] * P_A_coeffs[k][t]
                v_val = S2_V[t, s] * P_B_coeffs[k][t]
                
                if u_val != 0:
                    # Map 2x2 index t to 4x4 indices
                    # t=0 -> A00 (rb 0, cb 0)
                    # t=1 -> A01 (rb 0, cb 1)
                    # t=2 -> A10 (rb 1, cb 0)
                    # t=3 -> A11 (rb 1, cb 1)
                    rb = t // 2
                    cb = t % 2
                    idxs = get_2x2_indices_4x4(rb, cb)
                    for idx in idxs:
                        u_vec[idx] += u_val
                
                if v_val != 0:
                    rb = t // 2
                    cb = t % 2
                    idxs = get_2x2_indices_4x4(rb, cb)
                    for idx in idxs:
                        v_vec[idx] += v_val
            
            # W vector: S2_W[t, s] determines which 2x2 block of C gets the contribution.
            # The contribution is scaled by the "weight" of Pk in the final C formula.
            # Wait, the W vector in the tensor decomposition is part of the basis.
            # The final C is a linear combination of the Pk results.
            # C = w1*P1 + w2*P2 + ...
            # So the W vector for the term (k, s) is S2_W[:, s] * (coefficient of Pk in C).
            # But the coefficients of Pk in C depend on which block of C we are filling.
            # Actually, the Strassen decomposition of 4x4 is:
            # C = sum_k (linear_combination_of_Pk)
            # Each Pk is a 2x2 matrix.
            # So C00 = 1*P1_00 + 1*P4_00 - 1*P5_00 + 1*P7_00
            # C01 = 1*P3_01 + 1*P5_01
            # etc.
            
            # This means the W vector for the term (k, s) is not just S2_W.
            # It is S2_W[:, s] but placed in the specific 2x2 block of C determined by the formula.
            # AND scaled by the coefficient of Pk for that block.
            
            # Let's iterate over the 4 blocks of C (00, 01, 10, 11).
            # For each block, determine the coefficient of Pk.
            # C00: P1(1), P4(1), P5(-1), P7(1)
            # C01: P3(1), P5(1)
            # C10: P2(1), P4(1)
            # C11: P1(1), P2(-1), P3(1), P6(1)
            
            coeff_map = {
                0: {0: 1, 3: 1, 4: -1, 6: 1}, # C00
                1: {2: 1, 4: 1},               # C01
                2: {1: 1, 3: 1},               # C10
                3: {0: 1, 1: -1, 2: 1, 5: 1}  # C11
            }
            
            for c_block in range(4):
                coeff = coeff_map[c_block].get(k, 0)
                if coeff == 0:
                    continue
                # The S2_W vector tells us which 2x2 index within the Pk result is non-zero.
                # We need to add coeff * S2_W[:, s] to the w_vec at the indices of C-block c_block.
                rb = c_block // 2
                cb = c_block % 2
                idxs = get_2x2_indices_4x4(rb, cb)
                for t in range(4):
                    w_val = S2_W[t, s] * coeff
                    if w_val != 0:
                        w_vec[idxs[t]] += w_val
            
            # Add this term to the main U, V, W matrices
            if col < rank:
                U[0:16, col] = u_vec
                V[0:16, col] = v_vec
                W[0:16, col] = w_vec
                col += 1
                
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
            
    # 2. Cross terms
    # A44 b: i 0-3, j 0-3, k=4
    for i in range(4):
        for j in range(4):
            if col < rank:
                U[i*5 + j, col] = 1.0
                V[j*5 + 4, col] = 1.0
                W[4*5 + i, col] = 1.0
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # a c: i 0-3, j=4, k 0-3
    for i in range(4):
        for k in range(4):
            if col < rank:
                U[i*5 + 4, col] = 1.0
                V[4*5 + k, col] = 1.0
                W[k*5 + i, col] = 1.0
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # beta B44: i=4, j 0-3, k 0-3
    for j in range(4):
        for k in range(4):
            if col < rank:
                U[4*5 + j, col] = 1.0
                V[j*5 + k, col] = 1.0
                W[k*5 + 4, col] = 1.0
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # a delta: i 0-3, j=4, k=4
    for i in range(4):
        if col < rank:
            U[i*5 + 4, col] = 1.0
            V[4*5 + 4, col] = 1.0
            W[4*5 + i, col] = 1.0
            col += 1
        if time.time() > deadline:
            break

    # alpha c: i=4, j=4, k 0-3
    for k in range(4):
        if col < rank:
            U[4*5 + 4, col] = 1.0
            V[4*5 + k, col] = 1.0
            W[k*5 + 4, col] = 1.0
            col += 1
        if time.time() > deadline:
            break

    # beta b: i=4, j 0-3, k=4
    for j in range(4):
        if col < rank:
            U[4*5 + j, col] = 1.0
            V[j*5 + 4, col] = 1.0
            W[4*5 + 4, col] = 1.0
            col += 1
        if time.time() > deadline:
            break

    # alpha delta: i=4, j=4, k=4
    if col < rank:
        U[4*5 + 4, col] = 1.0
        V[4*5 + 4, col] = 1.0
        W[4*5 + 4, col] = 1.0
        col += 1

    # If col < rank, pad with zeros (should not happen if rank is exact)
    if col < rank:
        U[:, col:] = 0
        V[:, col:] = 0
        W[:, col:] = 0
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
