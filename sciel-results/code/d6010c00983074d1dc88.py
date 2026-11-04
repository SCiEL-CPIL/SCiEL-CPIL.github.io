import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank 98 decomposition.
    # This is based on two independent 4x4x4 multiplications.
    # Each 4x4x4 multiplication is decomposed using block Strassen (rank 49).
    
    # Step 1: Define the rank-7 decomposition for 2x2 matrices.
    # Indices i,j,k in {0,1}.
    # U is 4x7, V is 4x7, W is 4x7.
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    
    # Strassen terms:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We need to express each Mi as a sum of rank-1 tensors.
    # Actually, the Strassen algorithm computes 7 products.
    # Each product is a 2x2 matrix multiplication? No, the inputs are scalars (sums/diffs).
    # Wait, the inputs to the 7 multiplications are linear combinations of the elements of A and B.
    # Let X_r be the linear combination of A elements, Y_r be the linear combination of B elements.
    # Then Z_r = X_r * Y_r.
    # Then C is a linear combination of Z_r.
    
    # The tensor rank is the number of such Z_r terms.
    # For 2x2, rank is 7.
    
    # Let's define the coefficients for U, V, W for the 2x2 case.
    # U[i*2+j, r] is the coefficient of A[i,j] in X_r.
    # V[j*2+k, r] is the coefficient of B[j,k] in Y_r.
    # W[k*2+i, r] is the coefficient of Z_r in C[i,k].
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: X = A00+A11, Y = B00+B11
    # C00 += M1, C11 += M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: X = A10+A11, Y = B00
    # C10 += M2, C11 -= M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3: X = A00, Y = B01-B11
    # C01 += M3, C11 += M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4: X = A11, Y = B10-B00
    # C00 += M4, C10 += M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5: X = A00+A01, Y = B11
    # C00 -= M5, C01 += M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6: X = A10-A00, Y = B00+B01
    # C11 += M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7: X = A01-A11, Y = B10+B11
    # C00 += M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Now we have a rank-7 decomposition for 2x2.
    # We need to build a rank-49 decomposition for 4x4.
    # A 4x4 matrix can be viewed as 2x2 blocks of 2x2 matrices.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # The Strassen algorithm computes 7 products of 2x2 matrices.
    # Each 2x2 product has rank 7.
    # So we have 7 * 7 = 49 terms.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # The indices for 4x4:
    # A index: i*4+j. i in {0,1} (block row), j in {0,1} (block col)?
    # No, i in {0,1,2,3}, j in {0,1,2,3}.
    # Block (I, J) contains A[I*2+i, J*2+j].
    # So A[I*2+i, J*2+j] corresponds to local index (i, j) in block (I, J).
    # Global index: (I*2+i)*4 + (J*2+j) = I*8 + i*4 + J*2 + j.
    
    # For each of the 7 Strassen terms (r2 in 0..6):
    # We need to compute X_r2 * Y_r2 where X_r2 and Y_r2 are 2x2 matrices.
    # X_r2 is a linear combination of the 4 blocks of A.
    # Y_r2 is a linear combination of the 4 blocks of B.
    # Let X_r2 = sum_{I,J} cA[r2, I, J] A[I, J].
    # Let Y_r2 = sum_{K,L} cB[r2, K, L] B[K, L].
    # Then Z_r2 = X_r2 * Y_r2.
    # Z_r2 is a 2x2 matrix.
    # The elements of Z_r2 are linear combinations of the elements of C.
    # C[I, K] += sum_{J,L} (X_r2[I, J] * Y_r2[J, K])?
    # No, the Strassen formula gives C blocks as linear combinations of Z_r2 blocks.
    # C[I, K] = sum_{r2} sum_{J, L} coeff[r2, I, K, J, L] * (X_r2 * Y_r2)[J, L]?
    # Actually, the standard Strassen for blocks works as:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # ...
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_r2 is a 2x2 matrix product.
    # The rank of the product of two 2x2 matrices is 7.
    # So for each r2, we have 7 rank-1 terms.
    # Total 49 terms.
    
    # We need the coefficients cA, cB, and the combination coefficients for C.
    
    # cA for M1: A00 + A11. Blocks (0,0) and (1,1).
    # cA[r2, I, J] is 1 if block (I,J) is included.
    
    cA = np.zeros((7, 2, 2))
    cB = np.zeros((7, 2, 2))
    
    # M1
    cA[0, 0, 0] = 1; cA[0, 1, 1] = 1
    cB[0, 0, 0] = 1; cB[0, 1, 1] = 1
    
    # M2
    cA[1, 1, 0] = 1; cA[1, 1, 1] = 1
    cB[1, 0, 0] = 1
    
    # M3
    cA[2, 0, 0] = 1
    cB[2, 0, 1] = 1; cB[2, 1, 1] = -1
    
    # M4
    cA[3, 1, 1] = 1
    cB[3, 1, 0] = 1; cB[3, 0, 0] = -1
    
    # M5
    cA[4, 0, 0] = 1; cA[4, 0, 1] = 1
    cB[4, 1, 1] = 1
    
    # M6
    cA[5, 1, 0] = 1; cA[5, 0, 0] = -1
    cB[5, 0, 0] = 1; cB[5, 0, 1] = 1
    
    # M7
    cA[6, 0, 1] = 1; cA[6, 1, 1] = -1
    cB[6, 1, 0] = 1; cB[6, 1, 1] = 1
    
    # Coefficients for C blocks.
    # C[I, K] = sum_{r2} sum_{J, L} coeffC[r2, I, K, J, L] * (X_r2 * Y_r2)[J, L]
    # Actually, (X_r2 * Y_r2) is a 2x2 matrix. Let it be Z_r2.
    # C[I, K] is a 2x2 matrix.
    # C[I, K] = sum_{r2} coeffC[r2, I, K] * Z_r2?
    # No, Z_r2 is the full 2x2 product.
    # The formula is C[I, K] = sum_{r2} coeffC[r2, I, K] * Z_r2.
    
    coeffC = np.zeros((7, 2, 2))
    # C00
    coeffC[0, 0, 0] = 1
    coeffC[3, 0, 0] = 1
    coeffC[4, 0, 0] = -1
    coeffC[6, 0, 0] = 1
    # C01
    coeffC[2, 0, 1] = 1
    coeffC[4, 0, 1] = 1
    # C10
    coeffC[1, 1, 0] = 1
    coeffC[3, 1, 0] = 1
    # C11
    coeffC[0, 1, 1] = 1
    coeffC[1, 1, 1] = -1
    coeffC[2, 1, 1] = 1
    coeffC[5, 1, 1] = 1
    
    # Now construct U4, V4, W4.
    # For each r2 in 0..6:
    #   For each local rank term r_local in 0..6 (from U2, V2, W2):
    #     The global term index is r2 * 7 + r_local.
    #     U4[global_idx, r2*7+r_local] = sum_{I,J} cA[r2, I, J] * U2[(I*2+i_local)*2 + (J*2+j_local)? No.
    #     U2 index is (i_local*2 + j_local).
    #     The global A index for block (I,J) and local (i,j) is (I*2+i)*4 + (J*2+j).
    #     So U4[global_A_idx, r] = cA[r2, I, J] * U2[i*2+j, r_local].
    
    for r2 in range(7):
        for r_local in range(7):
            r_global = r2 * 7 + r_local
            for I in range(2):
                for J in range(2):
                    if cA[r2, I, J] != 0:
                        for i in range(2):
                            for j in range(2):
                                global_A_idx = (I*2+i)*4 + (J*2+j)
                                local_U_idx = i*2 + j
                                U4[global_A_idx, r_global] += cA[r2, I, J] * U2[local_U_idx, r_local]
                                
            for K in range(2):
                for L in range(2):
                    if cB[r2, K, L] != 0:
                        for j in range(2): # local row of B block? No, B is [K,L]. Local indices (j, k)
                            for k in range(2):
                                global_B_idx = (K*2+j)*4 + (L*2+k)
                                local_V_idx = j*2 + k
                                V4[global_B_idx, r_global] += cB[r2, K, L] * V2[local_V_idx, r_local]
                                
            for I in range(2):
                for K in range(2):
                    if coeffC[r2, I, K] != 0:
                        for i in range(2):
                            for k in range(2):
                                global_C_idx = (I*2+i)*4 + (K*2+k)
                                local_W_idx = k*2 + i
                                W4[global_C_idx, r_global] += coeffC[r2, I, K] * W2[local_W_idx, r_local]
                                
    # Now we have U4 (16x49), V4 (16x49), W4 (16x49) for 4x4x4.
    # We need 4x4x8.
    # Duplicate for k in 0..3 and k in 4..7.
    
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # Block 1: k in 0..3
    # U is same.
    U[:, 0:49] = U4
    # V indices: j*8 + k. k in 0..3.
    # V4 indices: j*4 + k_local.
    for j in range(4):
        for k_local in range(4):
            V[j*8 + k_local, 0:49] = V4[j*4 + k_local, 0:49]
    # W indices: k*4 + i. k in 0..3.
    # W4 indices: k_local*4 + i.
    for k_local in range(4):
        for i in range(4):
            W[k_local*4 + i, 0:49] = W4[k_local*4 + i, 0:49]
            
    # Block 2: k in 4..7
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            V[j*8 + (4 + k_local), 49:98] = V4[j*4 + k_local, 0:49]
    for k_local in range(4):
        for i in range(4):
            W[(4 + k_local)*4 + i, 49:98] = W4[k_local*4 + i, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
