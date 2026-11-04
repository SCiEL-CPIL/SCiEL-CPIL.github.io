import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrices
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    # r = 0..6
    
    # Define the 7 Strassen steps for 2x2
    # Each step is a tuple (u_coeff, v_coeff, w_coeff) for each element
    # But it's easier to just define the linear combinations
    
    # Let's generate the 2x2 Strassen decomposition
    # U2, V2, W2
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: (A00+A11)(B00+B11)
    # A00: U2[0,0]=1, A11: U2[3,0]=1
    # B00: V2[0,0]=1, B11: V2[3,0]=1
    # Output: C00 += M0, C11 += M0
    # W2 index k*2+i.
    # C00 is i=0, k=0 -> index 0.
    # C11 is i=1, k=1 -> index 1*2+1=3.
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: (A10+A11)B00
    # A10: U2[2,1]=1, A11: U2[3,1]=1
    # B00: V2[0,1]=1
    # Output: C10 += M1
    # C10 is i=1, k=0 -> index 0*2+1=1.
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1
    
    # r=2: A00(B01-B11)
    # A00: U2[0,2]=1
    # B01: V2[1,2]=1, B11: V2[3,2]=-1
    # Output: C01 += M2
    # C01 is i=0, k=1 -> index 1*2+0=2.
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1
    
    # r=3: A11(B10-B00)
    # A11: U2[3,3]=1
    # B10: V2[2,3]=1, B00: V2[0,3]=-1
    # Output: C11 += M3
    # C11 is i=1, k=1 -> index 3.
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[3][3] = 1
    
    # r=4: (A00+A01)B11
    # A00: U2[0,4]=1, A01: U2[1,4]=1
    # B11: V2[3,4]=1
    # Output: C01 -= M4, C11 -= M4
    # C01 index 2, C11 index 3.
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[2][4] = -1; W2[3][4] = -1
    
    # r=5: (A10-A00)(B00+B01)
    # A10: U2[2,5]=1, A00: U2[0,5]=-1
    # B00: V2[0,5]=1, B01: V2[1,5]=1
    # Output: C11 += M5
    # C11 index 3.
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # r=6: (A01-A11)(B10+B11)
    # A01: U2[1,6]=1, A11: U2[3,6]=-1
    # B10: V2[2,6]=1, B11: V2[3,6]=1
    # Output: C00 += M6
    # C00 index 0.
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Now combine for 4x4
    # Blocks A0, A1, A2, A3 correspond to 2x2 submatrices
    # A0 is top-left, A1 top-right, A2 bottom-left, A3 bottom-right
    # Indices for 4x4:
    # A0: i in 0,1; j in 0,1
    # A1: i in 0,1; j in 2,3
    # A2: i in 2,3; j in 0,1
    # A3: i in 2,3; j in 2,3
    
    # Strassen steps for blocks:
    # M1 = (A0+A3)(B0+B3)
    # M2 = (A2+A3)B0
    # M3 = A0(B1-B3)
    # M4 = A3(B2-B0)
    # M5 = (A0+A1)B3
    # M6 = (A2-A0)(B0+B1)
    # M7 = (A1-A3)(B2+B3)
    
    # Output blocks:
    # C0 = M1 + M4 - M5 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M2 + M3 + M6
    
    # For each block product Mk, we use the 2x2 Strassen factors.
    # The indices for U, V, W in 4x4 are derived from the block indices and the 2x2 indices.
    
    for block_r in range(7):
        # Determine the linear combination for U, V, W for this block product
        # We need to map the block indices to 4x4 indices.
        
        # For each of the 7 scalar Strassen steps within the block product
        for scalar_r in range(7):
            r = block_r * 7 + scalar_r
            
            # Get the 2x2 factors
            u2 = U2
            v2 = V2
            w2 = W2
            
            # Apply the block Strassen combination
            # This requires knowing which blocks are involved in Mk.
            
            # Let's define the block coefficients for U, V, W for each Mk.
            # Mk = Xk Yk
            # Xk is a linear combination of A blocks.
            # Yk is a linear combination of B blocks.
            
            # Coefficients for A blocks (U)
            # Coefficients for B blocks (V)
            # Coefficients for C blocks (W)
            
            # We can define the block factors similarly to U2, V2, W2 but for blocks.
            # Let U_block, V_block, W_block be 4x7 matrices for blocks.
            # U_block[block_idx, block_r]
            
            U_block = [[0]*7 for _ in range(4)]
            V_block = [[0]*7 for _ in range(4)]
            W_block = [[0]*7 for _ in range(4)]
            
            # r=0: (A0+A3)(B0+B3)
            U_block[0][0] = 1; U_block[3][0] = 1
            V_block[0][0] = 1; V_block[3][0] = 1
            W_block[0][0] = 1; W_block[3][0] = 1 # C0 += M0, C3 += M0
            
            # r=1: (A2+A3)B0
            U_block[2][1] = 1; U_block[3][1] = 1
            V_block[0][1] = 1
            W_block[2][1] = 1 # C2 += M1
            
            # r=2: A0(B1-B3)
            U_block[0][2] = 1
            V_block[1][2] = 1; V_block[3][2] = -1
            W_block[1][2] = 1 # C1 += M2
            
            # r=3: A3(B2-B0)
            U_block[3][3] = 1
            V_block[2][3] = 1; V_block[0][3] = -1
            W_block[0][3] = 1 # C0 += M3
            
            # r=4: (A0+A1)B3
            U_block[0][4] = 1; U_block[1][4] = 1
            V_block[3][4] = 1
            W_block[1][4] = -1 # C1 -= M4
            W_block[0][4] = 0 # Wait, C0 = M1 + M4 - M5 + M7.
            # Let's re-verify C0 formula.
            # C0 = M1 + M4 - M5 + M7
            # M4 is r=3. So C0 += M3 (block r=3).
            # M5 is r=4. So C0 -= M4 (block r=4).
            # So W_block[0][4] should be -1.
            W_block[0][4] = -1
            
            # r=5: (A2-A0)(B0+B1)
            U_block[2][5] = 1; U_block[0][5] = -1
            V_block[0][5] = 1; V_block[1][5] = 1
            W_block[3][5] = 1 # C3 += M5
            
            # r=6: (A1-A3)(B2+B3)
            U_block[1][6] = 1; U_block[3][6] = -1
            V_block[2][6] = 1; V_block[3][6] = 1
            W_block[0][6] = 1 # C0 += M6
            
            # Now combine block factors with 2x2 factors
            # For each non-zero entry in U_block, V_block, W_block
            # Multiply with the corresponding 2x2 factors
            
            # U_block[b_u, block_r] = c_u
            # V_block[b_v, block_r] = c_v
            # W_block[b_w, block_r] = c_w
            
            # For a specific scalar_r, we have U2[i2, scalar_r], V2[j2, scalar_r], W2[k2, scalar_r]
            
            # We need to sum over all block indices b_u, b_v, b_w
            # such that U_block[b_u, block_r] != 0, etc.
            
            # Actually, the product is:
            # U_4x4[i, r] = sum_{b_u, i2} U_block[b_u, block_r] * U2[i2, scalar_r] * (mapping i, i2, b_u)
            
            # Mapping:
            # Block b_u has rows i_b, cols j_b.
            # i = i_b * 2 + i2
            # j = j_b * 2 + j2 (for U)
            
            # Let's iterate over all i, j, k
            # This might be slow if done naively, but 16x16 is small.
            
            for i in range(4):
                for j in range(4):
                    val_u = 0
                    for b_u in range(4):
                        # Check if i, j belong to block b_u
                        # Block 0: i in 0,1; j in 0,1
                        # Block 1: i in 0,1; j in 2,3
                        # Block 2: i in 2,3; j in 0,1
                        # Block 3: i in 2,3; j in 2,3
                        if (b_u == 0 and i < 2 and j < 2) or \
                           (b_u == 1 and i < 2 and j >= 2) or \
                           (b_u == 2 and i >= 2 and j < 2) or \
                           (b_u == 3 and i >= 2 and j >= 2):
                            i2 = i % 2
                            j2 = j % 2
                            u2_val = U2[i2*2 + j2][scalar_r]
                            if u2_val != 0 and U_block[b_u][block_r] != 0:
                                val_u += U_block[b_u][block_r] * u2_val
                    U[i*4 + j][r] = val_u
                    
            for j in range(4):
                for k in range(4):
                    val_v = 0
                    for b_v in range(4):
                        if (b_v == 0 and j < 2 and k < 2) or \
                           (b_v == 1 and j < 2 and k >= 2) or \
                           (b_v == 2 and j >= 2 and k < 2) or \
                           (b_v == 3 and j >= 2 and k >= 2):
                            j2 = j % 2
                            k2 = k % 2
                            v2_val = V2[j2*2 + k2][scalar_r]
                            if v2_val != 0 and V_block[b_v][block_r] != 0:
                                val_v += V_block[b_v][block_r] * v2_val
                    V[j*4 + k][r] = val_v
                    
            for k in range(4):
                for i in range(4):
                    val_w = 0
                    for b_w in range(4):
                        if (b_w == 0 and k < 2 and i < 2) or \
                           (b_w == 1 and k < 2 and i >= 2) or \
                           (b_w == 2 and k >= 2 and i < 2) or \
                           (b_w == 3 and k >= 2 and i >= 2):
                            k2 = k % 2
                            i2 = i % 2
                            w2_val = W2[k2*2 + i2][scalar_r]
                            if w2_val != 0 and W_block[b_w][block_r] != 0:
                                val_w += W_block[b_w][block_r] * w2_val
                    W[k*4 + i][r] = val_w
                    
    return {"U": U, "V": V, "W": W}
