import numpy as np
import time

def construct():
    start_time = time.time()
    
    # 1. Strassen 2x2 decomposition
    # U, V, W for 2x2
    # Rank 7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1: (a+d)(e+h)
    # a=0, d=2, e=0, h=3
    U2[0, 0] = 1; U2[2, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C00 += M1, C11 += M1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (c+d)e
    # c=2, d=3, e=0
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    # C10 += M2, C11 += -M2
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3: a(f-h)
    # a=0, f=1, h=3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    # C01 += M3, C11 += M3
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4: d(g-f)
    # d=3, g=2, f=1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[1, 3] = -1
    # C00 += M4, C10 += M4
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5: (a+b)h
    # a=0, b=1, h=3
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    # C00 += -M5, C01 += M5
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6: (c-a)(e+f)
    # c=2, a=0, e=0, f=1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    # C11 += M6
    W2[3, 5] = 1
    
    # M7: (b-d)(g+h)
    # b=1, d=3, g=2, h=3
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    # C00 += M7
    W2[0, 6] = 1
    
    # Verify 2x2
    # (Optional)
    
    # 2. Construct 4x4 decomposition from 2x2 Strassen
    # Rank 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    # Block indices (bi, bj) correspond to 2x2 submatrices.
    # bi, bj in {0, 1}.
    # Element index in 4x4: bi*2 + i, bj*2 + j.
    
    # We iterate over the 7 Strassen terms for the block multiplication.
    # For each term t (0..6), we have 7 scalar multiplications (from the 2x2 Strassen inside).
    # Total 49 terms.
    
    for t in range(7):
        for s in range(7):
            c_idx = t * 7 + s
            
            # Determine which blocks are multiplied for term t
            # Based on Strassen formulas:
            # M1: A11+A22, B11+B22
            # M2: A21+A22, B11
            # M3: A11, B12-B22
            # M4: A22, B21-B12
            # M5: A11+A12, B22
            # M6: A21-A11, B11+B12
            # M7: A12-A22, B21+B22
            
            # We need the coefficients for U, V, W for the block level.
            # Let's define the block-level U, V, W for the 7 terms.
            # U_block[bi*2+bi2, t] ... this is getting complex.
            # Easier: Just expand the 2x2 Strassen logic to 4x4.
            
            # Let's define the block coefficients for the 7 terms.
            # U_block[bi, bj, t]
            # V_block[bi, bj, t]
            # W_block[bi, bj, t]
            
            # This is tedious to hardcode manually in the loop.
            # I will create a helper to generate the 4x4 decomposition.
            
            pass
            
    # Alternative: Just use the known fact that R(M4) = 49 and construct it by embedding.
    # I will implement the embedding logic.
    
    # For each of the 7 Strassen block terms:
    # We have a linear combination of A blocks and B blocks.
    # Then we multiply the resulting 2x2 blocks using the 2x2 Strassen decomposition.
    
    # Let's define the block coefficients for the 7 terms.
    # U_blk[bi, bj, t]
    # V_blk[bi, bj, t]
    # W_blk[bi, bj, t]
    
    U_blk = np.zeros((2, 2, 7))
    V_blk = np.zeros((2, 2, 7))
    W_blk = np.zeros((2, 2, 7))
    
    # M1: (A11+A22)(B11+B22)
    U_blk[0,0,0] = 1; U_blk[1,1,0] = 1
    V_blk[0,0,0] = 1; V_blk[1,1,0] = 1
    W_blk[0,0,0] = 1; W_blk[1,1,0] = 1 # C11 += M1, C22 += M1
    
    # M2: (A21+A22)B11
    U_blk[1,0,1] = 1; U_blk[1,1,1] = 1
    V_blk[0,0,1] = 1
    W_blk[1,0,1] = 1; W_blk[1,1,1] = -1 # C21 += M2, C22 += -M2
    
    # M3: A11(B12-B22)
    U_blk[0,0,2] = 1
    V_blk[0,1,2] = 1; V_blk[1,1,2] = -1
    W_blk[0,1,2] = 1; W_blk[1,1,2] = 1 # C12 += M3, C22 += M3
    
    # M4: A22(B21-B12)
    U_blk[1,1,3] = 1
    V_blk[1,0,3] = 1; V_blk[0,1,3] = -1
    W_blk[0,0,3] = 1; W_blk[1,0,3] = 1 # C11 += M4, C21 += M4
    
    # M5: (A11+A12)B22
    U_blk[0,0,4] = 1; U_blk[0,1,4] = 1
    V_blk[1,1,4] = 1
    W_blk[0,0,4] = -1; W_blk[0,1,4] = 1 # C11 += -M5, C12 += M5
    
    # M6: (A21-A11)(B11+B12)
    U_blk[1,0,5] = 1; U_blk[0,0,5] = -1
    V_blk[0,0,5] = 1; V_blk[0,1,5] = 1
    W_blk[1,1,5] = 1 # C22 += M6
    
    # M7: (A12-A22)(B21+B22)
    U_blk[0,1,6] = 1; U_blk[1,1,6] = -1
    V_blk[1,0,6] = 1; V_blk[1,1,6] = 1
    W_blk[0,0,6] = 1 # C11 += M7
    
    # Now combine with the 2x2 Strassen decomposition (U2, V2, W2)
    # The 4x4 indices:
    # A index: bi*2 + i, bj*2 + j
    # B index: bi*2 + i, bj*2 + j (Wait, B is 4x4, so indices are bi*2+i, bj*2+j)
    # C index: bi*2 + i, bj*2 + j
    
    # For each block term t and inner term s:
    for t in range(7):
        for s in range(7):
            c_idx = t * 7 + s
            
            # U4 coefficients
            for bi in range(2):
                for bj in range(2):
                    u_coeff = U_blk[bi, bj, t]
                    if u_coeff != 0:
                        # Add u_coeff * U2[i, s] to U4[bi*2+i, bj*2+j, c_idx]
                        for i in range(2):
                            for j in range(2):
                                U4[bi*2+i, bj*2+j, c_idx] += u_coeff * U2[i, s]
                                
            # V4 coefficients
            for bi in range(2):
                for bj in range(2):
                    v_coeff = V_blk[bi, bj, t]
                    if v_coeff != 0:
                        # Add v_coeff * V2[i, s] to V4[bi*2+i, bj*2+j, c_idx]
                        for i in range(2):
                            for j in range(2):
                                V4[bi*2+i, bj*2+j, c_idx] += v_coeff * V2[i, s]
                                
            # W4 coefficients
            for bi in range(2):
                for bj in range(2):
                    w_coeff = W_blk[bi, bj, t]
                    if w_coeff != 0:
                        # Add w_coeff * W2[i, s] to W4[bi*2+i, bj*2+j, c_idx]
                        for i in range(2):
                            for j in range(2):
                                W4[bi*2+i, bj*2+j, c_idx] += w_coeff * W2[i, s]
                                
    # 3. Construct 5x5 decomposition
    # Rank = 49 (for 4x4) + 1 + 16 + 4 + 4 + 4 + 4 + 1 = 83
    # We need to map the 4x4 decomposition and the small blocks into 5x5.
    
    U5 = np.zeros((25, 83))
    V5 = np.zeros((25, 83))
    W5 = np.zeros((25, 83))
    
    # 4x4 part (top-left)
    # Maps 4x4 indices to 5x5 indices (same)
    U5[0:16, 0:49] = U4
    V5[0:16, 0:49] = V4
    W5[0:16, 0:49] = W4
    
    # Next, the other blocks.
    # Block A41 (4x1) and B14 (1x4) -> C44 part
    # Rank 1.
    # U: 4x1 part of A. Indices 0..3 (col 0 of A? No, A is 5x5).
    # A indices: i in 0..3, j=4.
    # B indices: i=0, j in 0..3.
    # C indices: i in 0..3, j in 0..3.
    
    c_idx = 49
    # U5 for A41: i in 0..3, j=4. Index i*5+4.
    for i in range(4):
        U5[i*5+4, c_idx] = 1
    # V5 for B14: i=0, j in 0..3. Index 0*5+j = j.
    for j in range(4):
        V5[j, c_idx] = 1
    # W5 for C44: i in 0..3, j in 0..3. Index j*5+i.
    for i in range(4):
        for j in range(4):
            W5[j*5+i, c_idx] = 1
            
    c_idx += 1 # 50
    
    # A44 (4x4) and B41 (4x1) -> C41 part
    # Rank 16.
    # A44: i in 0..3, j in 0..3.
    # B41: i in 0..3, j=0.
    # C41: i in 0..3, j=0.
    # C_i0 = sum_j A_ij B_j0.
    # This is 16 multiplications.
    for i in range(4):
        for j in range(4):
            U5[i*5+j, c_idx] = 1
            V5[j*5+0, c_idx] = 1
            W5[0*5+i, c_idx] = 1
            c_idx += 1
            
    # c_idx should be 50 + 16 = 66
    
    # A41 (4x1) and B11 (1x1) -> C41 part (j=0)
    # Rank 4.
    # A41: i in 0..3, j=4.
    # B11: i=0, j=0.
    # C41: i in 0..3, j=0.
    # C_i0 += A_i4 B_00.
    for i in range(4):
        U5[i*5+4, c_idx] = 1
        V5[0*5+0, c_idx] = 1
        W5[0*5+i, c_idx] = 1
        c_idx += 1
        
    # c_idx should be 66 + 4 = 70
    
    # A14 (1x4) and B44 (4x4) -> C14 part
    # Rank 4.
    # A14: i=0, j in 0..3.
    # B44: i in 0..3, j in 0..3.
    # C14: i=0, j in 0..3.
    # C_0k = sum_j A_0j B_jk.
    for j in range(4):
        for k in range(4):
            U5[0*5+j, c_idx] = 1
            V5[j*5+k, c_idx] = 1
            W5[k*5+0, c_idx] = 1
            c_idx += 1
            
    # c_idx should be 70 + 16 = 86?
    # Wait, Rank of M_{1x4x4} is 4?
    # C_0k = sum_j A_0j B_jk.
    # This is a vector-matrix product.
    # It requires 4*4 = 16 multiplications if done naively.
    # Can we do it in rank 4?
    # T_{0, j, k} = 1.
    # Matricization: 1 x 4 x 4.
    # Rank is 4?
    # Let's check.
    # Sum_c U_{0, c} V_{j, c} W_{k, c} = 1.
    # If R=4, we can set U_{0, c} = 1, V_{j, c} = delta_{j,c}, W_{k, c} = delta_{k,c}?
    # Sum delta_{j,c} delta_{k,c} = delta_{jk}.
    # This gives the identity, not the product.
    # The product is C_{0k} = sum_j A_{0j} B_{jk}.
    # This is a sum of 16 terms.
    # The rank of M_{1x4x4} is 4? No, it's 4?
    # Actually, M_{1xnxn} has rank n?
    # Let's check n=2. M_{1x2x2}.
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # This is 4 multiplications.
    # Rank 4?
    # Yes, for 1xnxn, the rank is n^2.
    # So Rank of M_{1x4x4} is 16.
    
    # So my previous calculation was wrong.
    # Let's re-calculate the total rank.
    # 1. A44 B44: 49
    # 2. A41 B14: 1 (Rank 1 for 4x1x4? Yes, outer product of 4x1 and 1x4 is rank 1 tensor?
    #    C_ij = A_i0 B_0j.
    #    T_{i,0,j} = 1.
    #    Sum U_i V_0 W_j = 1.
    #    U_i=1, V_0=1, W_j=1. Rank 1. Correct.)
    # 3. A44 B41: 16 (Rank 16 for 4x4x1?
    #    C_i0 = sum_j A_ij B_j0.
    #    T_{i,j,0} = 1.
    #    Sum U_{ij} V_j W_i = 1.
    #    This is 16 independent terms. Rank 16. Correct.)
    # 4. A41 B11: 4 (Rank 4 for 4x1x1?
    #    C_i0 = A_i0 B_00.
    #    T_{i,0,0} = 1.
    #    Sum U_i V_0 W_i = 1.
    #    U_i=1, V_0=1, W_i=1. Rank 1?
    #    Wait. C_i0 = A_i0 B_00.
    #    This is 4 separate multiplications A_i0 * B_00.
    #    But B_00 is the same for all i.
    #    Can we share the multiplication?
    #    No, because A_i0 are different.
    #    But the tensor T_{i,0,0} = 1.
    #    Sum_c U_{i,c} V_{0,c} W_{0,c} = 1.
    #    If R=1, U_i=1, V_0=1, W_0=1.
    #    Then C_i0 = 1 * A_i0 * B_00 * 1 = A_i0 B_00.
    #    So Rank is 1?
    #    Yes, because the middle index is fixed.
    #    So Rank of M_{4x1x1} is 1.
    #    Similarly M_{1x1x4} is 1.
    #    M_{1x4x1} is 1.
    
    # Let's re-verify ranks of mixed blocks.
    # M_{m x 1 x p}: Rank 1.
    # M_{m x n x 1}: Rank mn.
    # M_{1 x n x p}: Rank np.
    
    # So:
    # 1. A44 B44: 49
    # 2. A41 B14: M_{4x1x4}. Rank 1.
    # 3. A44 B41: M_{4x4x1}. Rank 16.
    # 4. A41 B11: M_{4x1x1}. Rank 1.
    # 5. A14 B44: M_{1x4x4}. Rank 16.
    # 6. A11 B14: M_{1x1x4}. Rank 1.
    # 7. A14 B41: M_{1x4x1}. Rank 4.
    # 8. A11 B11: M_{1x1x1}. Rank 1.
    
    # Total Rank = 49 + 1 + 16 + 1 + 16 + 1 + 4 + 1 = 89.
    
    # My previous code allocated 83.
    # I need to adjust the allocation.
    
    # Let's restart the 5x5 construction with correct ranks.
    
    U5 = np.zeros((25, 89))
    V5 = np.zeros((25, 89))
    W5 = np.zeros((25, 89))
    
    # 1. 4x4 part (0..48)
    U5[0:16, 0:49] = U4
    V5[0:16, 0:49] = V4
    W5[0:16, 0:49] = W4
    
    c_idx = 49
    
    # 2. A41 B14 (Rank 1)
    # A: i 0..3, j=4. B: i=0, j 0..3. C: i 0..3, j 0..3.
    for i in range(4):
        U5[i*5+4, c_idx] = 1
    for j in range(4):
        V5[0*5+j, c_idx] = 1
    for i in range(4):
        for j in range(4):
            W5[j*5+i, c_idx] = 1
    c_idx += 1
    
    # 3. A44 B41 (Rank 16)
    # A: i 0..3, j 0..3. B: i 0..3, j=0. C: i 0..3, j=0.
    for i in range(4):
        for j in range(4):
            U5[i*5+j, c_idx] = 1
            V5[j*5+0, c_idx] = 1
            W5[0*5+i, c_idx] = 1
            c_idx += 1
            
    # 4. A41 B11 (Rank 1)
    # A: i 0..3, j=4. B: i=0, j=0. C: i 0..3, j=0.
    for i in range(4):
        U5[i*5+4, c_idx] = 1
    V5[0*5+0, c_idx] = 1
    for i in range(4):
        W5[0*5+i, c_idx] = 1
    c_idx += 1
    
    # 5. A14 B44 (Rank 16)
    # A: i=0, j 0..3. B: i 0..3, j 0..3. C: i=0, j 0..3.
    for j in range(4):
        for k in range(4):
            U5[0*5+j, c_idx] = 1
            V5[j*5+k, c_idx] = 1
            W5[k*5+0, c_idx] = 1
            c_idx += 1
            
    # 6. A11 B14 (Rank 1)
    # A: i=0, j=0. B: i=0, j 0..3. C: i=0, j 0..3.
    U5[0*5+0, c_idx] = 1
    for j in range(4):
        V5[0*5+j, c_idx] = 1
    for j in range(4):
        W5[j*5+0, c_idx] = 1
    c_idx += 1
    
    # 7. A14 B41 (Rank 4)
    # A: i=0, j 0..3. B: i 0..3, j=0. C: i=0, j=0.
    for j in range(4):
        U5[0*5+j, c_idx] = 1
        V5[j*5+0, c_idx] = 1
        W5[0*5+0, c_idx] = 1
        c_idx += 1
        
    # 8. A11 B11 (Rank 1)
    # A: i=0, j=0. B: i=0, j=0. C: i=0, j=0.
    U5[0*5+0, c_idx] = 1
    V5[0*5+0, c_idx] = 1
    W5[0*5+0, c_idx] = 1
    c_idx += 1
    
    # Total rank 89.
    
    return {'U': U5.tolist(), 'V': V5.tolist(), 'W': W5.tolist()}

