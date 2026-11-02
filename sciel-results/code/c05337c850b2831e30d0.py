import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    # Split A (2x4) into A1 (2x2), A2 (2x2)
    # Split B (4x6) into B1 (2x6), B2 (2x6)
    # Result C = A1*B1 + A2*B2
    # Each A_i (2x2) * B_i (2x6) can be done by splitting B_i into three 2x2 blocks
    # B_i = [B_i1, B_i2, B_i3]
    # C_i = A_i * B_i = [A_i*B_i1, A_i*B_i2, A_i*B_i3]
    # Each A_i*B_ij is a 2x2x2 multiplication, rank 7 (Strassen)
    # Total rank = 2 * 3 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8x42
    V = [[0] * rank for _ in range(m * p)] # 24x42
    W = [[0] * rank for _ in range(p * n)] # 12x42
    
    # Helper to add a Strassen block
    # Inputs:
    # a, b, c, d (indices in U for A)
    # x, y, z, w (indices in V for B)
    # Output indices in W
    # Returns the next column index
    col = 0
    
    def add_strassen_block(a_idx, b_idx, c_idx, d_idx, x_idx, y_idx, z_idx, w_idx, out_idx_11, out_idx_12, out_idx_21, out_idx_22):
        nonlocal col
        # 1. (a+d)(x+z)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[x_idx][col] = 1; V[z_idx][col] = 1
        W[out_idx_11][col] = 1; W[out_idx_22][col] = 1
        col += 1
        
        # 2. (c+d)(y-z)  -- Wait, standard Strassen:
        # M1 = (a+d)(x+z)
        # M2 = (c+d)z
        # M3 = a(y-z)
        # M4 = d(x-y)
        # M5 = (a+b)x
        # M6 = (c+d)(y-x)
        # M7 = (a+b)(z+y)
        
        # Let's use standard Strassen formulas for 2x2:
        # A = [[a, b], [c, d]], B = [[x, y], [z, w]]
        # Note: My V indices x, y, z, w correspond to B elements.
        # B = [[x, y], [z, w]]
        
        # M1 = (a+d)(x+z)
        # Already did M1 above? No, I wrote (x+z) for V.
        # Let's restart the block helper with correct indices.
        
        pass

    # Correct Strassen Block
    def add_strassen(a_idx, b_idx, c_idx, d_idx, x_idx, y_idx, z_idx, w_idx, w11_idx, w12_idx, w21_idx, w22_idx):
        nonlocal col
        # M1 = (a+d)(x+z)
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[x_idx][col] = 1; V[z_idx][col] = 1
        W[w11_idx][col] = 1; W[w22_idx][col] = 1
        col += 1
        
        # M2 = (c+d)z
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[z_idx][col] = 1
        W[w21_idx][col] = 1; W[w22_idx][col] = -1
        col += 1
        
        # M3 = a(y-z)
        U[a_idx][col] = 1
        V[y_idx][col] = 1; V[z_idx][col] = -1
        W[w12_idx][col] = 1; W[w22_idx][col] = -1
        col += 1
        
        # M4 = d(x-y)
        U[d_idx][col] = 1
        V[x_idx][col] = 1; V[y_idx][col] = -1
        W[w11_idx][col] = 1; W[w22_idx][col] = -1
        col += 1
        
        # M5 = (a+b)x
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[x_idx][col] = 1
        W[w11_idx][col] = -1; W[w12_idx][col] = 1
        col += 1
        
        # M6 = (c+d)(y-x)
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[y_idx][col] = 1; V[x_idx][col] = -1
        W[w21_idx][col] = 1; W[w22_idx][col] = -1
        col += 1
        
        # M7 = (a+b)(z+y)
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[z_idx][col] = 1; V[y_idx][col] = 1
        W[w12_idx][col] = 1; W[w22_idx][col] = 1
        col += 1

    # Map indices
    # A is 2x4. Flat index: i*m + j.
    # A1 (2x2): indices 0, 1, 2, 3. (a00, a01, a10, a11)
    # A2 (2x2): indices 4, 5, 6, 7.
    
    # B is 4x6. Flat index: j*p + k.
    # B1 (2x6): rows 0,1. Indices 0..5, 6..11.
    # B2 (2x6): rows 2,3. Indices 12..17, 18..23.
    
    # W is 6x2. Flat index: k*n + i.
    # C = A*B. C is 2x6.
    # C1 (from A1*B1) is 2x6.
    # C2 (from A2*B2) is 2x6.
    # C = C1 + C2.
    
    # W indices for C1:
    # C1 is 2x6.
    # Row 0 of C1: k=0..5, i=0. W index: k*2 + 0.
    # Row 1 of C1: k=0..5, i=1. W index: k*2 + 1.
    
    # W indices for C2:
    # Same indices, added to.
    
    # Process A1 * B1
    # A1: a=0, b=1, c=2, d=3
    # B1: 
    # Block 1 (cols 0,1): x=0, y=1, z=6, w=7
    # Block 2 (cols 2,3): x=2, y=3, z=8, w=9
    # Block 3 (cols 4,5): x=4, y=5, z=10, w=11
    
    # Outputs for Block 1:
    # C1[0,0] -> W[0]
    # C1[0,1] -> W[1]
    # C1[1,0] -> W[2]
    # C1[1,1] -> W[3]
    
    add_strassen(0, 1, 2, 3, 0, 1, 6, 7, 0, 1, 2, 3)
    add_strassen(0, 1, 2, 3, 2, 3, 8, 9, 4, 5, 6, 7)
    add_strassen(0, 1, 2, 3, 4, 5, 10, 11, 8, 9, 10, 11)
    
    # Process A2 * B2
    # A2: a=4, b=5, c=6, d=7
    # B2:
    # Block 1 (cols 0,1): x=12, y=13, z=18, w=19
    # Block 2 (cols 2,3): x=14, y=15, z=20, w=21
    # Block 3 (cols 4,5): x=16, y=17, z=22, w=23
    
    # Outputs added to same W indices
    add_strassen(4, 5, 6, 7, 12, 13, 18, 19, 0, 1, 2, 3)
    add_strassen(4, 5, 6, 7, 14, 15, 20, 21, 4, 5, 6, 7)
    add_strassen(4, 5, 6, 7, 16, 17, 22, 23, 8, 9, 10, 11)
    
    return {"U": U, "V": V, "W": W}
