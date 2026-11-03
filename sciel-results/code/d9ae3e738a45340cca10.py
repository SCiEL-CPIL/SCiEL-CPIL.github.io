import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2():
    # Returns U, V, W for 2x2 matrix multiplication
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: A: 00,01,10,11 -> 0,1,2,3
    #          B: 00,01,10,11 -> 0,1,2,3
    #          C: 00,01,10,11 -> 0,1,2,3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C10, C11
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[2, 2] = 1 # C01, C10
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1 # C00, C01
    
    return U, V, W

def _compose_strassen_4x4():
    # Compose 2x2 Strassen into 4x4 Strassen
    # Rank 7 * 7 = 49
    U2, V2, W2 = _strassen_2x2()
    
    # 4x4 indices:
    # A: 4x4 = 16
    # B: 4x4 = 16
    # C: 4x4 = 16
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block indices for 4x4:
    # A00: 0-3, A01: 4-7, A10: 8-11, A11: 12-15
    # B00: 0-3, B01: 4-7, B10: 8-11, B11: 12-15
    # C00: 0-3, C01: 4-7, C10: 8-11, C11: 12-15
    
    # We iterate over the 7 high-level Strassen terms
    # Each term is a product of two 2x2 matrices
    # U_high[i, c_high] gives the coefficient for A_block
    # V_high[j, c_high] gives the coefficient for B_block
    # W_high[k, c_high] gives the coefficient for C_block
    
    # But Strassen is recursive.
    # M_c = (A_u)(B_v) where A_u is a linear combo of blocks.
    # We need to map the 2x2 Strassen U,V,W to the 4x4 indices.
    
    # For each high-level term c (0..6):
    # The "A" part is a linear combination of 2x2 blocks of A.
    # U2[i, c] is the coefficient for A_block_i in the 2x2 Strassen.
    # A_block_i corresponds to a 2x2 submatrix of A4.
    # We need to expand A_block_i into the 16 entries of A4.
    
    # Block mapping:
    # Block 0 (00): indices 0,1,4,5? No.
    # A4 is row-major.
    # A4[0,0]=0, A4[0,1]=1, A4[0,2]=2, A4[0,3]=3
    # A4[1,0]=4, A4[1,1]=5, A4[1,2]=6, A4[1,3]=7
    # A4[2,0]=8, A4[2,1]=9, A4[2,2]=10, A4[2,3]=11
    # A4[3,0]=12, A4[3,1]=13, A4[3,2]=14, A4[3,3]=15
    
    # Block 00: 0,1,4,5
    # Block 01: 2,3,6,7
    # Block 10: 8,9,12,13
    # Block 11: 10,11,14,15
    
    block_indices = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    for c_high in range(7):
        # Determine which blocks of A and B are involved in this high-level term
        # U2[i, c_high] is non-zero for some i
        # V2[j, c_high] is non-zero for some j
        
        # The term is (Sum_i U2[i,c] A_i) * (Sum_j V2[j,c] B_j)
        # This is a product of two 2x2 matrices.
        # We use the 2x2 Strassen decomposition for this product.
        # Let M_sub = A_sub * B_sub.
        # M_sub has rank 7.
        # The coefficients for the 4x4 matrices will be the product of the high-level coeffs and the low-level coeffs.
        
        # For each low-level term c_low (0..6):
        # The contribution to U4 is:
        # For each i (block of A), if U2[i, c_high] != 0:
        #   For each local index li in block i:
        #     U4[global_index(li), c_high*7+c_low] += U2[i, c_high] * U2[li, c_low]
        
        # Similarly for V and W.
        
        for c_low in range(7):
            c_global = c_high * 7 + c_low
            
            # U
            for i_block in range(4):
                coeff_high = U2[i_block, c_high]
                if coeff_high == 0: continue
                for li in range(4):
                    coeff_low = U2[li, c_low]
                    if coeff_low == 0: continue
                    global_idx = block_indices[i_block][li]
                    U4[global_idx, c_global] += coeff_high * coeff_low
            
            # V
            for j_block in range(4):
                coeff_high = V2[j_block, c_high]
                if coeff_high == 0: continue
                for lj in range(4):
                    coeff_low = V2[lj, c_low]
                    if coeff_low == 0: continue
                    global_idx = block_indices[j_block][lj]
                    V4[global_idx, c_global] += coeff_high * coeff_low
            
            # W
            for k_block in range(4):
                coeff_high = W2[k_block, c_high]
                if coeff_high == 0: continue
                for lk in range(4):
                    coeff_low = W2[lk, c_low]
                    if coeff_low == 0: continue
                    global_idx = block_indices[k_block][lk]
                    W4[global_idx, c_global] += coeff_high * coeff_low
                    
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    
    # Part 1: M4,4,4 (Strassen) Rank 49
    U4, V4, W4 = _compose_strassen_4x4()
    
    # Part 2: M4,4,1 (Schoolbook) Rank 16
    # A is 4x4, B is 4x1, C is 4x1
    # U: (16, 16), V: (4, 16), W: (4, 16)
    U16 = np.zeros((16, 16))
    V16 = np.zeros((4, 16))
    W16 = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U16[i * 4 + j, c] = 1
            V16[j, c] = 1
            W16[i, c] = 1 # C is 4x1, index i
            
    # Part 3: M4,1,5 (Schoolbook) Rank 20
    # A is 4x1, B is 1x5, C is 4x5
    # U: (4, 20), V: (5, 20), W: (20, 20)
    U20 = np.zeros((4, 20))
    V20 = np.zeros((5, 20))
    W20 = np.zeros((20, 20))
    for i in range(4):
        for k in range(5):
            c = i * 5 + k
            U20[i, c] = 1
            V20[k, c] = 1
            W20[i * 5 + k, c] = 1
            
    # Combine into final U, V, W
    # U: (20, 85), V: (25, 85), W: (20, 85)
    Rank = 49 + 16 + 20
    U = np.zeros((20, Rank))
    V = np.zeros((25, Rank))
    W = np.zeros((20, Rank))
    
    # Fill Part 1 (cols 0-48)
    # U4 is (16, 49). Maps to U[0:16, 0:49]
    # But U indices are i*5+j.
    # A1 is A[0:4, 0:4].
    # U4[i*4+j, c] corresponds to A[i, j].
    # In final U, this is index i*5 + j.
    for i in range(4):
        for j in range(4):
            idx_final = i * 5 + j
            idx_src = i * 4 + j
            U[idx_final, 0:49] = U4[idx_src, 0:49]
            
    # V4 is (16, 49). Maps to V[0:16, 0:49]
    # B1 is B[0:4, 0:4].
    # V4[j*4+k, c] corresponds to B[j, k].
    # In final V, this is index j*5 + k.
    for j in range(4):
        for k in range(4):
            idx_final = j * 5 + k
            idx_src = j * 4 + k
            V[idx_final, 0:49] = V4[idx_src, 0:49]
            
    # W4 is (16, 49). Maps to W[0:16, 0:49]
    # C1 is C[0:4, 0:4].
    # W4[k*4+i, c] corresponds to C[k, i]?
    # Wait, W index is k*n+i.
    # In 4x4x4, n=4. So index k*4+i.
    # In final W, n=4. So index k*4+i.
    # So W[0:16, 0:49] = W4[0:16, 0:49] directly?
    # W4 indices are k*4+i.
    # Final W indices are k*4+i.
    # Yes.
    W[0:16, 0:49] = W4[0:16, 0:49]
    
    # Fill Part 2 (cols 49-64)
    # U16 is (16, 16).
    # A1 is A[0:4, 0:4].
    # U16[i*4+j, c] -> U[i*5+j, 49+c]
    for i in range(4):
        for j in range(4):
            idx_final = i * 5 + j
            idx_src = i * 4 + j
            U[idx_final, 49:65] = U16[idx_src, 0:16]
            
    # V16 is (4, 16).
    # b2 is B[0:4, 4].
    # V16[j, c] -> V[j*5+4, 49+c]
    for j in range(4):
        idx_final = j * 5 + 4
        idx_src = j
        V[idx_final, 49:65] = V16[idx_src, 0:16]
        
    # W16 is (4, 16).
    # C12 is C[0:4, 4].
    # W16[i, c] -> W[4*4+i, 49+c]
    for i in range(4):
        idx_final = 4 * 4 + i
        idx_src = i
        W[idx_final, 49:65] = W16[idx_src, 0:16]
        
    # Fill Part 3 (cols 65-84)
    # U20 is (4, 20).
    # A2 is A[0:4, 4].
    # U20[i, c] -> U[i*5+4, 65+c]
    for i in range(4):
        idx_final = i * 5 + 4
        idx_src = i
        U[idx_final, 65:85] = U20[idx_src, 0:20]
        
    # V20 is (5, 20).
    # B2 is B[4, 0:5].
    # V20[k, c] -> V[4*5+k, 65+c]
    for k in range(5):
        idx_final = 4 * 5 + k
        idx_src = k
        V[idx_final, 65:85] = V20[idx_src, 0:20]
        
    # W20 is (20, 20).
    # C2 is C[0:4, 0:5].
    # W20[k*4+i, c] -> W[k*4+i, 65+c]
    W[0:20, 65:85] = W20[0:20, 0:20]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
