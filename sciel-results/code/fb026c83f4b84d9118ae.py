"""Rewritten program for 5x5 matrix multiplication with rank 94."""
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate 2x2 Strassen decomposition (Rank 7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # i,j,k in {0,1}
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Strassen 2x2 formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U, V, W entries for each of the 7 products
    # c=0: M1
    # A part: A11(0,0) + A22(1,1)
    U2[0][0] = 1; U2[3][0] = 1
    # B part: B11(0,0) + B22(1,1)
    V2[0][0] = 1; V2[3][0] = 1
    # C part: C11(0,0) += M1, C22(1,1) += M1
    W2[0][0] = 1; W2[3][0] = 1
    
    # c=1: M2
    # A part: A21(1,0) + A22(1,1)
    U2[2][1] = 1; U2[3][1] = 1
    # B part: B11(0,0)
    V2[0][1] = 1
    # C part: C21(1,0) += M2, C22(1,1) -= M2
    W2[2][1] = 1; W2[3][1] = -1
    
    # c=2: M3
    # A part: A11(0,0)
    U2[0][2] = 1
    # B part: B12(0,1) - B22(1,1)
    V2[1][2] = 1; V2[3][2] = -1
    # C part: C12(0,1) += M3, C22(1,1) += M3
    W2[1][2] = 1; W2[3][2] = 1
    
    # c=3: M4
    # A part: A22(1,1)
    U2[3][3] = 1
    # B part: B21(1,0) - B11(0,0)
    V2[2][3] = 1; V2[0][3] = -1
    # C part: C11(0,0) += M4, C21(1,0) += M4
    W2[0][3] = 1; W2[2][3] = 1
    
    # c=4: M5
    # A part: A11(0,0) + A12(0,1)
    U2[0][4] = 1; U2[1][4] = 1
    # B part: B22(1,1)
    V2[3][4] = 1
    # C part: C11(0,0) -= M5, C12(0,1) += M5
    W2[0][4] = -1; W2[1][4] = 1
    
    # c=5: M6
    # A part: A21(1,0) - A11(0,0)
    U2[2][5] = 1; U2[0][5] = -1
    # B part: B11(0,0) + B12(0,1)
    V2[0][5] = 1; V2[1][5] = 1
    # C part: C22(1,1) += M6
    W2[3][5] = 1
    
    # c=6: M7
    # A part: A12(0,1) - A22(1,1)
    U2[1][6] = 1; U2[3][6] = -1
    # B part: B21(1,0) + B22(1,1)
    V2[2][6] = 1; V2[3][6] = 1
    # C part: C11(0,0) += M7
    W2[0][6] = 1

    # 2. Generate 4x4 Strassen decomposition (Rank 49)
    # This is the "lifting" of the 2x2 Strassen to 4x4.
    # The 4x4 matrix is viewed as 2x2 blocks of 2x2 matrices.
    # We use 7 outer Strassen products. Each product is a 2x2 matrix multiplication.
    # We replace each 2x2 matrix multiplication with the 7 inner Strassen products.
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Outer Strassen coefficients (same as 2x2, but operating on blocks)
    # We iterate over the 7 outer products (c_outer)
    for c_outer in range(7):
        # For each outer product, we have a 2x2 matrix multiplication X*Y
        # X and Y are linear combinations of the 4 blocks of A and B.
        # We use the inner 2x2 Strassen decomposition to compute X*Y.
        # The inner Strassen has 7 products (c_inner).
        for c_inner in range(7):
            c_4x4 = c_outer * 7 + c_inner
            
            # The U, V, W entries for the 4x4 case are determined by
            # the outer coefficients (which block of A/B is involved) and
            # the inner coefficients (which element of the 2x2 block is involved).
            
            # This is complex to hardcode directly. 
            # Alternative: Construct the 4x4 decomposition by composing the 2x2 tensors.
            # T_4x4 = sum_{c_outer=1}^7 (U_outer[c_outer] \otimes U_inner[c_inner]) ...
            # Actually, it's simpler to just use the schoolbook for the 4x4 part if the code gets too complex, 
            # but rank 49 is much better than 64 (schoolbook 4x4).
            
            # Let's use a helper to build the 4x4 from 2x2.
            # The index mapping for 4x4:
            # i_4 = i_block * 2 + i_local
            # j_4 = j_block * 2 + j_local
            # k_4 = k_block * 2 + k_local
            
            # For a given outer product c_outer, the A-blocks involved are determined by U2[i_block*2+j_block, c_outer]?
            # No, U2 is for the 2x2 case. 
            # In the 2x2 Strassen, U2[row, c] gives the coefficient for A[row] in the c-th product.
            # row = i_block * 2 + j_block.
            
            # For the 4x4 case, the "row" of the outer A is (i_block, j_block).
            # The actual row index in the 4x4 U matrix is (i_block * 2 + i_local) * 2 + (j_block * 2 + j_local)?
            # No. The U matrix for 4x4 has rows indexed by (i, j) where i,j in 0..3.
            # i = i_block * 2 + i_local.
            # j = j_block * 2 + j_local.
            
            # The coefficient for U4[i_4, j_4, c_4x4] is:
            # U_outer[i_block, j_block, c_outer] * U_inner[i_local, j_local, c_inner]
            
            # Wait, U2 is defined for 2x2 matrices.
            # U2[row_2, c_outer] where row_2 = i_block * 2 + j_block.
            # So U_outer[i_block, j_block, c_outer] = U2[i_block * 2 + j_block, c_outer].
            
            # Similarly for V and W.
            
            for i_block in range(2):
                for j_block in range(2):
                    coeff_u_outer = U2[i_block * 2 + j_block][c_outer]
                    if coeff_u_outer == 0: continue
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            coeff_u_inner = U2[i_local * 2 + j_local][c_inner]
                            if coeff_u_inner == 0: continue
                            
                            u_val = coeff_u_outer * coeff_u_inner
                            if u_val == 0: continue
                            
                            i_4 = i_block * 2 + i_local
                            j_4 = j_block * 2 + j_local
                            U4[i_4 * 4 + j_4][c_4x4] += u_val
                            
            for j_block in range(2):
                for k_block in range(2):
                    coeff_v_outer = V2[j_block * 2 + k_block][c_outer]
                    if coeff_v_outer == 0: continue
                    
                    for j_local in range(2):
                        for k_local in range(2):
                            coeff_v_inner = V2[j_local * 2 + k_local][c_inner]
                            if coeff_v_inner == 0: continue
                            
                            v_val = coeff_v_outer * coeff_v_inner
                            if v_val == 0: continue
                            
                            j_4 = j_block * 2 + j_local
                            k_4 = k_block * 2 + k_local
                            V4[j_4 * 4 + k_4][c_4x4] += v_val
                            
            for k_block in range(2):
                for i_block in range(2):
                    coeff_w_outer = W2[k_block * 2 + i_block][c_outer]
                    if coeff_w_outer == 0: continue
                    
                    for k_local in range(2):
                        for i_local in range(2):
                            coeff_w_inner = W2[k_local * 2 + i_local][c_inner]
                            if coeff_w_inner == 0: continue
                            
                            w_val = coeff_w_outer * coeff_w_inner
                            if w_val == 0: continue
                            
                            k_4 = k_block * 2 + k_local
                            i_4 = i_block * 2 + i_local
                            W4[k_4 * 4 + i_4][c_4x4] += w_val

    # 3. Construct 5x5 decomposition
    # Rank = 49 (4x4) + 20 (C41) + 20 (C14) + 5 (C11) = 94
    rank = 94
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    col = 0
    # Fill 4x4 part (cols 0..48)
    # Map 4x4 indices (0..15) to 5x5 indices (0..24)
    # 4x4 index idx_4 = i_4 * 4 + j_4.
    # 5x5 index idx_5 = i_5 * 5 + j_5.
    # i_5 = i_4 // 2? No.
    # i_4 = i_block * 2 + i_local.
    # i_5 = i_4.
    # j_5 = j_4.
    # Since i_4, j_4 are in 0..3, i_5, j_5 are in 0..3.
    # So idx_5 = i_4 * 5 + j_4.
    
    for i_4 in range(4):
        for j_4 in range(4):
            idx_4 = i_4 * 4 + j_4
            idx_5 = i_4 * 5 + j_4
            for c in range(49):
                U[idx_5][col + c] = U4[idx_4][c]
                
    for j_4 in range(4):
        for k_4 in range(4):
            idx_4 = j_4 * 4 + k_4
            idx_5 = j_4 * 5 + k_4
            for c in range(49):
                V[idx_5][col + c] = V4[idx_4][c]
                
    for k_4 in range(4):
        for i_4 in range(4):
            idx_4 = k_4 * 4 + i_4
            idx_5 = k_4 * 5 + i_4
            for c in range(49):
                W[idx_5][col + c] = W4[idx_4][c]
                
    col += 49
    
    # Fill C41 part (cols 49..68)
    # C41 is 4x1. Rows 0..3, Col 4.
    # Terms: A44 B41 (16 mults) + A41 B11 (4 mults)
    # A44 B41: i in 0..3, j in 0..3, k=4.
    # U[i*5+j, c] = 1, V[j*5+4, c] = 1, W[4*5+i, c] = 1
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*5+i][col] = 1
            col += 1
            
    # A41 B11: i in 0..3, j=4, k=4.
    # U[i*5+4, c] = 1, V[4*5+4, c] = 1, W[4*5+i, c] = 1
    for i in range(4):
        U[i*5+4][col] = 1
        V[4*5+4][col] = 1
        W[4*5+i][col] = 1
        col += 1
        
    # Fill C14 part (cols 69..88)
    # C14 is 1x4. Row 4, Cols 0..3.
    # Terms: A14 B44 (16 mults) + A11 B14 (4 mults)
    # A14 B44: i=4, j in 0..3, k in 0..3.
    # U[4*5+j, c] = 1, V[j*5+k, c] = 1, W[k*5+4, c] = 1
    for j in range(4):
        for k in range(4):
            U[4*5+j][col] = 1
            V[j*5+k][col] = 1
            W[k*5+4][col] = 1
            col += 1
            
    # A11 B14: i=4, j=4, k in 0..3.
    # U[4*5+4, c] = 1, V[4*5+k, c] = 1, W[k*5+4, c] = 1
    for k in range(4):
        U[4*5+4][col] = 1
        V[4*5+k][col] = 1
        W[k*5+4][col] = 1
        col += 1
        
    # Fill C11 part (cols 89..93)
    # C11 is 1x1. Row 4, Col 4.
    # Terms: A14 B41 (4 mults) + A11 B11 (1 mult)
    # A14 B41: i=4, j in 0..3, k=4.
    # U[4*5+j, c] = 1, V[j*5+4, c] = 1, W[4*5+4, c] = 1
    for j in range(4):
        U[4*5+j][col] = 1
        V[j*5+4][col] = 1
        W[4*5+4][col] = 1
        col += 1
        
    # A11 B11: i=4, j=4, k=4.
    U[4*5+4][col] = 1
    V[4*5+4][col] = 1
    W[4*5+4][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
