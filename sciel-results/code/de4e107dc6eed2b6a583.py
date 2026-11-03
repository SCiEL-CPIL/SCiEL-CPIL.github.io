import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct the decomposition using the Kronecker product of Strassen's algorithm for 2x2 matrices.
    # Strassen's algorithm for 2x2 matrix multiplication has rank 7.
    # The 4x4 matrix multiplication can be viewed as block multiplication of 2x2 matrices.
    # The rank is 7 * 7 = 49.
    
    # Strassen's 7 terms for 2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We define the vectors for the 2x2 case.
    # Indices for 2x2:
    # U_idx: i*2 + j
    # V_idx: j*2 + k
    # W_idx: k*2 + i
    
    # The tensor T_2x2 has T[i,j,k] = 1.
    # We need U, V, W of size 4x7 such that sum U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    
    # Strassen coefficients:
    # The bilinear forms are linear combinations of the products.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M1 - M2 + M6
    # C21 = M2 + M4 - M1 + M3
    # C22 = M1 - M4 + M2 - M6
    
    # Let's define the 7 rank-1 tensors for 2x2.
    # Each term Mr corresponds to a product of linear forms in A and B.
    # L1_A = a+d, L1_B = e+h
    # L2_A = c+d, L2_B = e
    # L3_A = a, L3_B = f-h
    # L4_A = d, L4_B = g-e
    # L5_A = a+b, L5_B = h
    # L6_A = c-a, L6_B = e+f
    # L7_A = b-d, L7_B = g+h
    
    # The output C is linear combination of Ms.
    # C11 = 1*M1 + 0*M2 + 0*M3 + 1*M4 - 1*M5 + 0*M6 + 1*M7
    # C12 = 0*M1 - 1*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7
    # C21 = 0*M1 + 1*M2 + 1*M3 - 1*M4 + 0*M5 + 0*M6 + 0*M7
    # C22 = 1*M1 + 1*M2 + 0*M3 - 1*M4 + 0*M5 - 1*M6 + 0*M7
    
    # We can represent this as U, V, W matrices of size 4x7.
    # U[i,j,r] is the coefficient of A_ij in Lr_A.
    # V[j,k,r] is the coefficient of B_jk in Lr_B.
    # W[k,i,r] is the coefficient of Mr in C_ki (note: C11 is k=0,i=0? No. C_{ik}. 
    # Prompt W index is k*n+i. For 2x2, n=2. Index k*2+i.
    # C00 (i=0,k=0) -> W[0*2+0, r] = W[0, r]
    # C01 (i=0,k=1) -> W[1*2+0, r] = W[2, r]
    # C10 (i=1,k=0) -> W[0*2+1, r] = W[1, r]
    # C11 (i=1,k=1) -> W[1*2+1, r] = W[3, r]
    
    # Let's build U2, V2, W2 for 2x2.
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*2+i)
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1: (a+d)(e+h)
    U2[0, 0] = 1 # a
    U2[3, 0] = 1 # d
    V2[0, 0] = 1 # e
    V2[3, 0] = 1 # h
    
    # M2: (c+d)e
    U2[2, 1] = 1 # c
    U2[3, 1] = 1 # d
    V2[0, 1] = 1 # e
    
    # M3: a(f-h)
    U2[0, 2] = 1 # a
    V2[1, 2] = 1 # f
    V2[3, 2] = -1 # h
    
    # M4: d(g-e)
    U2[3, 3] = 1 # d
    V2[2, 3] = 1 # g
    V2[0, 3] = -1 # e
    
    # M5: (a+b)h
    U2[0, 4] = 1 # a
    U2[1, 4] = 1 # b
    V2[3, 4] = 1 # h
    
    # M6: (c-a)(e+f)
    U2[2, 5] = 1 # c
    U2[0, 5] = -1 # a
    V2[0, 5] = 1 # e
    V2[1, 5] = 1 # f
    
    # M7: (b-d)(g+h)
    U2[1, 6] = 1 # b
    U2[3, 6] = -1 # d
    V2[2, 6] = 1 # g
    V2[3, 6] = 1 # h
    
    # W2: Coefficients of Mr in C_ki
    # C00 (i=0, k=0) -> Index 0
    W2[0, 0] = 1 # M1
    W2[0, 3] = 1 # M4
    W2[0, 4] = -1 # M5
    W2[0, 6] = 1 # M7
    
    # C01 (i=0, k=1) -> Index 2 (k*2+i = 1*2+0)
    W2[2, 2] = 1 # M3
    W2[2, 0] = 1 # M1
    W2[2, 1] = -1 # M2
    W2[2, 5] = 1 # M6
    
    # C10 (i=1, k=0) -> Index 1 (k*2+i = 0*2+1)
    W2[1, 1] = 1 # M2
    W2[1, 3] = 1 # M4
    W2[1, 0] = -1 # M1
    W2[1, 2] = 1 # M3
    
    # C11 (i=1, k=1) -> Index 3 (k*2+i = 1*2+1)
    W2[3, 0] = 1 # M1
    W2[3, 3] = -1 # M4
    W2[3, 1] = 1 # M2
    W2[3, 5] = -1 # M6
    
    # Now construct 4x4 decomposition using Kronecker product.
    # The 4x4 matrix is composed of 2x2 blocks.
    # A_4 = [[A00, A01], [A10, A11]]
    # The indices for A_4 are (I, J) where I, J in {0,1} are block indices, and i, j in {0,1} are internal indices.
    # Global index for A: (I*2 + i) * 2 + (J*2 + j) = I*4 + i*2 + J*2 + j?
    # Prompt U index: i*m + j. Here m=4.
    # Let global row i_g = I*2 + i, global col j_g = J*2 + j.
    # Index = i_g * 4 + j_g = (I*2 + i)*4 + (J*2 + j) = I*8 + i*4 + J*2 + j.
    
    # Similarly for B: j_g = J*2 + j, k_g = K*2 + k.
    # Index = j_g * 4 + k_g = (J*2 + j)*4 + (K*2 + k) = J*8 + j*4 + K*2 + k.
    
    # For C: k_g = K*2 + k, i_g = I*2 + i.
    # Prompt W index: k*n + i. Here n=4.
    # Index = k_g * 4 + i_g = (K*2 + k)*4 + (I*2 + i) = K*8 + k*4 + I*2 + i.
    
    # The decomposition is the Kronecker product of the 2x2 decompositions.
    # Rank = 7 * 7 = 49.
    # The terms are indexed by (r1, r2) where r1, r2 in 0..6.
    # Total rank R = 49.
    
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # U contribution
            # U_global[i_g, j_g, r] = U2[i, j, r1] * U2[i', j', r2] ?
            # The block structure:
            # A_{I,J} is a 2x2 matrix.
            # The linear form L_r1 acts on the "block" indices? No.
            # The Kronecker product of bilinear forms:
            # L_{r1} \otimes L_{r2} (A) = sum_{I,J,i,j} U2[I,i,r1] U2[J,j,r2] A_{I*2+i, J*2+j}
            # Wait, the 2x2 Strassen acts on the elements of the 2x2 blocks.
            # So we need to apply the 2x2 decomposition to each 2x2 block of the 4x4 matrix?
            # No, the 4x4 multiplication is a single tensor.
            # The Kronecker product construction works as follows:
            # If T = sum u_r v_r w_r, then T \otimes T = sum (u_r \otimes u_s) (v_r \otimes v_s) (w_r \otimes w_s).
            # The tensor for 4x4 multiplication is the Kronecker product of the tensor for 2x2 multiplication.
            # T_4 = T_2 \otimes T_2.
            
            # So U_4 = U_2 \otimes U_2?
            # Let's check the indices.
            # U_2 has indices (i,j) for 2x2.
            # U_4 has indices (i_g, j_g) for 4x4.
            # i_g = I*2 + i, j_g = J*2 + j.
            # The Kronecker product of vectors u (size 4) and v (size 4) is size 16.
            # (u \otimes v)_{i*4+j} = u_i v_j.
            # Here, the index for U_2 is i*2+j.
            # So we need to map the 4x4 index to the Kronecker product of two 2x2 indices.
            
            # Let's construct U4 by iterating over the 16 indices.
            for i_g in range(4):
                for j_g in range(4):
                    # Decompose global indices into block and local
                    I = i_g // 2
                    i = i_g % 2
                    J = j_g // 2
                    j = j_g % 2
                    
                    # The term in U4 for index (i_g, j_g) and rank r=(r1, r2)
                    # comes from U2[I, J? No.
                    # The tensor T_2 has indices (i,j) for A, (j,k) for B, (k,i) for C.
                    # The Kronecker product T_4 has indices (I,i, J,j) for A?
                    # Actually, the 4x4 matrix A can be seen as A \in M_2(M_2).
                    # The tensor T_4 = T_2 \otimes T_2.
                    # The first factor of T_4 is A \otimes A? No.
                    # The first factor of T_2 is A \in M_{2,2}.
                    # The first factor of T_4 is A \in M_{4,4}.
                    # M_{4,4} \cong M_{2,2} \otimes M_{2,2}.
                    # So the first factor is the tensor product of the first factor of T_2 with itself.
                    
                    # U2 has shape (4, 7). Indices (i*2+j, r).
                    # We need to form U4 of shape (16, 49).
                    # U4[i_g*4+j_g, r1*7+r2] = U2[I*2+i? No.
                    
                    # Let's use the property:
                    # (u \otimes v)_{(i1, i2)} = u_{i1} v_{i2}.
                    # Here the index space of A is $M_{2,2} \otimes M_{2,2}$.
                    # Basis element $E_{I,J} \otimes E_{i,j}$ corresponds to $A_{I*2+i, J*2+j}$.
                    # The vector in $M_{2,2} \otimes M_{2,2}$ is $u_1 \otimes u_2$.
                    # The component at $(I,J) \otimes (i,j)$ is $u_1[I,J] u_2[i,j]$.
                    # Wait, U2 is a vector in $M_{2,2}^*$.
                    # So U4 is a vector in $(M_{2,2} \otimes M_{2,2})^* \cong M_{2,2}^* \otimes M_{2,2}^*$.
                    # So U4 = U2 \otimes U2.
                    
                    # Index mapping for U2: idx2 = i*2+j.
                    # Index mapping for U4: idx4 = i_g*4+j_g.
                    # i_g = I*2+i, j_g = J*2+j.
                    # idx4 = (I*2+i)*4 + (J*2+j) = I*8 + i*4 + J*2 + j.
                    
                    # We need U4[idx4, r1*7+r2] = U2[I*2+J? No.
                    # The first factor of U2 corresponds to the "outer" 2x2?
                    # Let's assume the Kronecker product structure matches the block structure.
                    # The "outer" indices are I, J. The "inner" indices are i, j.
                    # U2 acts on the 2x2 matrix.
                    # So U4 = U2 \otimes U2.
                    # The index of U2 is (row, col).
                    # In the Kronecker product, the index is (row1, col1, row2, col2).
                    # We need to map (I, J, i, j) to the flattened index.
                    # Standard Kronecker index: (row1 * dim2 + row2) * dim2 * dim2 + ...?
                    # Let's just compute it directly.
                    
                    # U2 index for (I, J) is I*2+J.
                    # U2 index for (i, j) is i*2+j.
                    
                    # Does U4[idx4, r] = U2[I*2+J, r1] * U2[i*2+j, r2]?
                    # Let's verify the tensor product.
                    # T_4 = sum_{r1,r2} (U2_r1 \otimes U2_r2) \otimes (V2_r1 \otimes V2_r2) \otimes (W2_r1 \otimes W2_r2).
                    # This matches the Kronecker product of the tensors.
                    
                    U[idx4, r] = U2[I*2+J, r1] * U2[i*2+j, r2]
                    
            # Similarly for V and W
            for j_g in range(4):
                for k_g in range(4):
                    J = j_g // 2
                    j = j_g % 2
                    K = k_g // 2
                    k = k_g % 2
                    idx4 = j_g * 4 + k_g
                    V[idx4, r] = V2[J*2+K, r1] * V2[j*2+k, r2]
            
            for k_g in range(4):
                for i_g in range(4):
                    K = k_g // 2
                    k = k_g % 2
                    I = i_g // 2
                    i = i_g % 2
                    idx4 = k_g * 4 + i_g
                    W[idx4, r] = W2[K*2+I, r1] * W2[k*2+i, r2]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
