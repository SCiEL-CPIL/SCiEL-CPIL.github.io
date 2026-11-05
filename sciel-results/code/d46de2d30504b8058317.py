import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define Strassen 2x2x2 decomposition
    # U, V, W are 4x7 matrices
    # Indices: i, j in 0..1 -> row i*2+j
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 1: (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 2: (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # Term 3: A00(B10-B11)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # Term 4: A11(B01-B00)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Term 5: (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Construct 4x4x4 decomposition (Rank 49)
    # U4, V4, W4 are 16x49
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # Iterate over block indices I, J, K
            # and local indices i, j, k
            # Note: Strassen terms define non-zero entries.
            # We can just iterate all and multiply.
            
            # For U:
            for I in range(2):
                for J in range(2):
                    u_block_val = U2[I*2 + J][r1] # Wait, U2 index is i*2+j. Here I,J are block indices.
                    # In Strassen 2x2, the "scalar" A_IJ corresponds to U2[I*2+J]?
                    # No, U2 is for 2x2 matrices. The indices of U2 are 0..3.
                    # The block A_IJ is a 2x2 matrix.
                    # The coefficient for the block A_IJ in the r1-th Strassen term is U2[I*2+J][r1]?
                    # Let's check.
                    # In Strassen, M1 = (A00+A11)(B00+B11).
                    # A00 is block (0,0). A11 is block (1,1).
                    # So yes, the coefficient for block (I,J) is U2[I*2+J][r1].
                    
                    if u_block_val == 0:
                        continue
                        
                    for i in range(2):
                        for j in range(2):
                            u_local_val = U2[i*2 + j][r2]
                            if u_local_val == 0:
                                continue
                            
                            # Global indices
                            gi = I * 2 + i
                            gj = J * 2 + j
                            row_u = gi * 4 + gj
                            
                            U4[row_u][r] += u_block_val * u_local_val
                            
            # For V:
            for J in range(2):
                for K in range(2):
                    v_block_val = V2[J*2 + K][r1]
                    if v_block_val == 0:
                        continue
                    
                    for j in range(2):
                        for k in range(2):
                            v_local_val = V2[j*2 + k][r2]
                            if v_local_val == 0:
                                continue
                            
                            gj = J * 2 + j
                            gk = K * 2 + k
                            row_v = gj * 4 + gk
                            
                            V4[row_v][r] += v_block_val * v_local_val
                            
            # For W:
            for K in range(2):
                for I in range(2):
                    w_block_val = W2[K*2 + I][r1]
                    if w_block_val == 0:
                        continue
                    
                    for k in range(2):
                        for i in range(2):
                            w_local_val = W2[k*2 + i][r2]
                            if w_local_val == 0:
                                continue
                            
                            gk = K * 2 + k
                            gi = I * 2 + i
                            row_w = gk * 4 + gi
                            
                            W4[row_w][r] += w_block_val * w_local_val

    # Construct 4x4x8 decomposition (Rank 98)
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(32)]
    
    # First half (k=0..3)
    for r in range(49):
        for row in range(16):
            U[row][r] = U4[row][r]
            V[row][r] = V4[row][r]
        for row in range(16):
            W[row][r] = W4[row][r]
            
    # Second half (k=4..7)
    for r in range(49):
        r_new = 49 + r
        # U is same
        for row in range(16):
            U[row][r_new] = U4[row][r]
            
        # V is shifted
        for row in range(16):
            # Find j, k from row: row = j*4 + k
            j = row // 4
            k = row % 4
            if V4[row][r] != 0:
                # New k is k + 4
                # New row is j*8 + (k+4)
                new_row = j * 8 + (k + 4)
                V[new_row][r_new] = V4[row][r]
                
        # W is shifted
        for row in range(16):
            # Find k, i from row: row = k*4 + i
            k = row // 4
            i = row % 4
            if W4[row][r] != 0:
                # New k is k + 4
                # New row is (k+4)*4 + i
                new_row = (k + 4) * 4 + i
                W[new_row][r_new] = W4[row][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
