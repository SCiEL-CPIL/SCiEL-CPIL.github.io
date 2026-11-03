import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Part 2: 4x4x1 using Schoolbook (Rank 16)
    
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- 4x4x4 Part ---
    # We need to construct the 49 vectors.
    # Strassen 2x2x2 vectors
    # u, v, w are lists of 7 vectors of length 4
    
    # Define Strassen 2x2x2 decomposition
    # Indices for 2x2 matrix: 00, 01, 10, 11 -> 0, 1, 2, 3
    # A: [[A00, A01], [A10, A11]]
    # B: [[B00, B01], [B10, B11]]
    # C: [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    u_2x2 = []
    v_2x2 = []
    w_2x2 = []
    
    # M1
    u = [1, 0, 0, 1] # A00, A11
    v = [1, 0, 0, 1] # B00, B11
    w = [1, 0, 0, 1] # C00, C11 (coeff 1)
    u_2x2.append(u); v_2x2.append(v); w_2x2.append(w)
    
    # M2
    u = [0, 0, 1, 1] # A10, A11
    v = [1, 0, 0, 0] # B00
    w = [0, 0, 1, 0] # C10 (coeff 1)
    u_2x2.append(u); v_2x2.append(v); w_2x2.append(w)
    
    # M3
    u = [1, 0, 0, 0] # A00
    v = [0, 1, 0, -1] # B01, -B11
    w = [0, 1, 0, 0] # C01 (coeff 1)
    u_2x2.append(u); v_2x2.append(v); w_2x2.append(w)
    
    # M4
    u = [0, 0, 0, 1] # A11
    v = [0, 0, 1, -1] # B10, -B00
    w = [-1, 0, 1, 0] # -C00, C10
    u_2x2.append(u); v_2x2.append(v); w_2x2.append(w)
    
    # M5
    u = [1, 1, 0, 0] # A00, A01
    v = [0, 0, 0, 1] # B11
    w = [1, 0, 0, 0] # C00
    u_2x2.append(u); v_2x2.append(v); w_2x2.append(w)
    
    # M6
    u = [0, 0, 1, -1] # A10, -A00
    v = [1, 1, 0, 0] # B00, B01
    w = [0, 0, 0, 1] # C11
    u_2x2.append(u); v_2x2.append(v); w_2x2.append(w)
    
    # M7
    u = [0, 1, 0, -1] # A01, -A11
    v = [0, 0, 1, 1] # B10, B11
    w = [-1, 1, 0, 1] # -C00, C01, C11
    u_2x2.append(u); v_2x2.append(v); w_2x2.append(w)
    
    # Now construct 4x4x4 from 2x2x2
    # A is 4x4, blocks A00, A01, A10, A11 (each 2x2)
    # Indices in A:
    # A00: 0,1,2,3
    # A01: 4,5,6,7
    # A10: 8,9,10,11
    # A11: 12,13,14,15
    
    # B is 4x4 (for this part), blocks B00, B01, B10, B11
    # Indices in B:
    # B00: 0,1,2,3
    # B01: 4,5,6,7
    # B10: 8,9,10,11
    # B11: 12,13,14,15
    
    # C is 4x4, blocks C00, C01, C10, C11
    # Indices in C:
    # C00: 0,1,2,3
    # C01: 4,5,6,7
    # C10: 8,9,10,11
    # C11: 12,13,14,15
    
    col = 0
    for r in range(7):
        # This corresponds to one of the 7 Strassen products of 2x2 blocks
        # u_2x2[r] gives the linear combo of A blocks
        # v_2x2[r] gives the linear combo of B blocks
        # w_2x2[r] gives the linear combo of C blocks
        
        # We need to expand this to 4x4 level
        # The 2x2 multiplication is applied to each pair of 2x2 blocks?
        # No, the Strassen algorithm for 4x4 uses 7 products of 2x2 matrices.
        # Each 2x2 matrix multiplication is decomposed into 7 scalar multiplications.
        # So for each of the 7 Strassen steps, we have a 2x2x2 tensor.
        # We iterate over the 7 inner Strassen steps.
        
        for inner_r in range(7):
            # The outer r determines which blocks of A and B are used
            # The inner inner_r determines the scalar multiplication within the 2x2 block
            
            # u_2x2[r] has entries for A00, A01, A10, A11
            # Let a_coeff = u_2x2[r]
            # We need to form the U vector for the 4x4 matrix
            # For the block A_k (where k is 0..3), if a_coeff[k] is non-zero,
            # we add a_coeff[k] * u_2x2[inner_r] to the corresponding part of U
            
            # Similarly for V and W
            
            U_vec = [0] * 16
            V_vec = [0] * 16
            W_vec = [0] * 16
            
            # Map 2x2 block index to 4x4 index offset
            # Block 0 (00): 0..3
            # Block 1 (01): 4..7
            # Block 2 (10): 8..11
            # Block 3 (11): 12..15
            
            for k in range(4):
                coeff_u = u_2x2[r][k]
                coeff_v = v_2x2[r][k]
                coeff_w = w_2x2[r][k]
                
                if coeff_u != 0:
                    for j in range(4):
                        U_vec[k*4 + j] += coeff_u * u_2x2[inner_r][j]
                        
                if coeff_v != 0:
                    for j in range(4):
                        V_vec[k*4 + j] += coeff_v * v_2x2[inner_r][j]
                        
                if coeff_w != 0:
                    for j in range(4):
                        W_vec[k*4 + j] += coeff_w * w_2x2[inner_r][j]
                        
            # Assign to U, V, W matrices
            for i in range(16):
                U[i][col] = U_vec[i]
                V[i][col] = V_vec[i]
                W[i][col] = W_vec[i]
            col += 1
            
    # --- 4x4x1 Part ---
    # Schoolbook for 4x4x1
    # A is 4x4 (indices 0..15)
    # B is 4x1 (indices 0..3 in the 4x5 matrix, i.e., column 4)
    # C is 1x4 (indices 0..3 in the 5x4 matrix, i.e., row 4)
    
    # B index in full V matrix: m*p = 4*5 = 20.
    # The 4x1 part uses B_{j, 4} for j=0..3.
    # V index = j * 5 + 4.
    
    # C index in full W matrix: p*n = 5*4 = 20.
    # The 1x4 part uses C_{4, i} for i=0..3.
    # W index = 4 * 4 + i = 16 + i.
    
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[16 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
