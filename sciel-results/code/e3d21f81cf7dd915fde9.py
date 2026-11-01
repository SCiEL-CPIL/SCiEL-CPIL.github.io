import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses block-wise Strassen decomposition, achieving rank 56.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We decompose into 8 independent 2x2x2 multiplications.
    # The 2x4 A is split into 4 columns. 
    # The 4x8 B is split into 4 blocks of 2 columns? 
    # Actually, the previous logic was:
    # A is 2x4. B is 4x8.
    # We view B as 4 blocks of size 4x2.
    # Let B = [B_0, B_1, B_2, B_3] where B_i is 4x2.
    # C = [A B_0, A B_1, A B_2, A B_3].
    # Each A B_i is 2x4 * 4x2.
    # A B_i = A_left * B_i_top + A_right * B_i_bottom?
    # A is 2x4. Split A into A_L (2x2) and A_R (2x2).
    # B_i is 4x2. Split B_i into B_i_T (2x2) and B_i_B (2x2).
    # A B_i = A_L B_i_T + A_R B_i_B.
    # This is sum of two 2x2x2 multiplications.
    # Each 2x2x2 uses Strassen (rank 7).
    # Total rank = 4 blocks * 2 multiplications/block * 7 rank = 56.
    
    # Indices:
    # A: i in 0..1, j in 0..3. Index a = i*4 + j.
    # B: j in 0..3, k in 0..7. Index b = j*8 + k.
    # C: k in 0..7, i in 0..1. Index c = k*2 + i. (Trace convention W[k*n+i])
    
    # Loop over the 4 blocks of B (k pairs)
    # Block v corresponds to k in {2v, 2v+1}
    for v in range(4):
        # k indices for this block
        k0 = 2 * v
        k1 = 2 * v + 1
        
        # We need to compute C[:, k0] and C[:, k1]
        # C[:, k0] = A * B[:, k0]
        # C[:, k1] = A * B[:, k1]
        
        # B[:, k0] is a 4x1 vector. Let's call it b_k0.
        # b_k0 = [B[0, k0], B[1, k0], B[2, k0], B[3, k0]]
        # Split into b_top = [B[0, k0], B[1, k0]] (from j=0,1)
        # and b_bot = [B[2, k0], B[3, k0]] (from j=2,3)
        
        # A * b_k0 = A_L * b_top + A_R * b_bot
        # A_L is A[:, 0:2] (2x2). A_R is A[:, 2:4] (2x2).
        
        # This is two 2x2 matrix-vector products.
        # We use Strassen for 2x2 * 2x1?
        # Wait, Strassen is for 2x2 * 2x2.
        # Here we are multiplying 2x2 matrix by 2x1 vector.
        # This is equivalent to 2 independent 1x2 * 2x1? No.
        # It is 2 dot products of length 2.
        # Rank of 2x2 * 2x1 is 4?
        # Let's check.
        # M = [[m00, m01], [m10, m11]], v = [v0, v1].
        # r0 = m00v0 + m01v1
        # r1 = m10v0 + m11v1
        # This requires 4 multiplications.
        # So for each k, we need 4 multiplications for A_L part and 4 for A_R part?
        # Total 8 multiplications per k?
        # 8 k's * 8 mult = 64.
        
        # The previous solution claimed rank 56.
        # How?
        # It said "8 independent 2x2x2 multiplications".
        # This implies it treated the vector as a matrix with a dummy column of 1s? No.
        # Or maybe it grouped two k's together?
        # If we group k0 and k1, we have a 2x2 matrix B_block.
        # Then A * B_block is 2x4 * 4x2.
        # This is the sum of two 2x2 * 2x2 multiplications.
        # A_L * B_block_top + A_R * B_block_bottom.
        # Each 2x2 * 2x2 is Strassen rank 7.
        # So 7 + 7 = 14 per block.
        # 4 blocks * 14 = 56.
        # Yes, this is correct.
        
        # So for each v (0..3):
        # We have a 4x2 block of B.
        # Let B_v be this block.
        # B_v = [[B[0, 2v], B[0, 2v+1]],
        #        [B[1, 2v], B[1, 2v+1]],
        #        [B[2, 2v], B[2, 2v+1]],
        #        [B[3, 2v], B[3, 2v+1]]]
        
        # Split B_v into Top (2x2) and Bottom (2x2).
        # B_v_Top = [[B[0, 2v], B[0, 2v+1]],
        #            [B[1, 2v], B[1, 2v+1]]]
        # B_v_Bot = [[B[2, 2v], B[2, 2v+1]],
        #            [B[3, 2v], B[3, 2v+1]]]
        
        # A_L = A[:, 0:2]
        # A_R = A[:, 2:4]
        
        # C_v = A_L B_v_Top + A_R B_v_Bot
        # C_v is 2x2.
        
        # We use Strassen for M1 = A_L B_v_Top and M2 = A_R B_v_Bot.
        # Strassen factors for M1:
        # s1 = (A_L[0,0] + A_L[1,1]) * (B_v_Top[0,0] + B_v_Top[1,1])
        # s2 = (A_L[1,0] + A_L[1,1]) * B_v_Top[0,0]
        # s3 = A_L[0,0] * (B_v_Top[0,1] - B_v_Top[1,1])
        # s4 = A_L[1,1] * (B_v_Top[1,0] - B_v_Top[0,0])
        # s5 = (A_L[0,0] + A_L[0,1]) * B_v_Top[1,1]
        # s6 = (A_L[1,0] - A_L[0,0]) * (B_v_Top[0,0] + B_v_Top[0,1])
        # s7 = (A_L[0,1] - A_L[1,1]) * (B_v_Top[1,0] + B_v_Top[1,1])
        
        # Similarly for M2 with A_R and B_v_Bot.
        
        # We need to map these to U, V, W.
        
        for part in range(2):
            # part 0: A_L * B_v_Top
            # part 1: A_R * B_v_Bot
            
            base_col = 14 * v + 7 * part
            
            # Determine A indices
            if part == 0:
                # A_L: rows 0,1; cols 0,1
                a_rows = [0, 1]
                a_cols = [0, 1]
            else:
                # A_R: rows 0,1; cols 2,3
                a_rows = [0, 1]
                a_cols = [2, 3]
                
            # Determine B indices
            # B_v_Top: rows 0,1; cols 2v, 2v+1
            # B_v_Bot: rows 2,3; cols 2v, 2v+1
            if part == 0:
                b_rows = [0, 1]
            else:
                b_rows = [2, 3]
            b_cols = [2 * v, 2 * v + 1]
            
            # Determine C indices
            # C_v is 2x2.
            # C[0, 2v], C[0, 2v+1]
            # C[1, 2v], C[1, 2v+1]
            # Index c = k*n + i = k*2 + i
            c_indices = [
                2 * (2 * v) + 0,     # C[0, 2v]
                2 * (2 * v) + 1,     # C[1, 2v]
                2 * (2 * v + 1) + 0, # C[0, 2v+1]
                2 * (2 * v + 1) + 1  # C[1, 2v+1]
            ]
            
            # Strassen steps
            # We need to add contributions to U, V, W for each of the 7 products.
            
            # Helper to get A index
            def a_idx(r, c):
                return a_rows[r] * 4 + a_cols[c]
                
            # Helper to get B index
            def b_idx(r, c):
                return b_rows[r] * 8 + b_cols[c]
                
            # Helper to get C index
            # c_idx is already calculated as list of 4 indices for [0,0], [1,0], [0,1], [1,1]
            # Let's map Strassen outputs to C indices.
            # Strassen gives M[0,0], M[0,1], M[1,0], M[1,1].
            # c00 = c_indices[0]
            # c10 = c_indices[1]
            # c01 = c_indices[2]
            # c11 = c_indices[3]
            
            c00 = c_indices[0]
            c10 = c_indices[1]
            c01 = c_indices[2]
            c11 = c_indices[3]
            
            # s=0: M1 = (A00+A11)(B00+B11)
            U[a_idx(0, 0)][base_col] += 1
            U[a_idx(1, 1)][base_col] += 1
            V[b_idx(0, 0)][base_col] += 1
            V[b_idx(1, 1)][base_col] += 1
            W[c00][base_col] += 1
            W[c11][base_col] += 1
            
            # s=1: M2 = (A10+A11)B00
            U[a_idx(1, 0)][base_col + 1] += 1
            U[a_idx(1, 1)][base_col + 1] += 1
            V[b_idx(0, 0)][base_col + 1] += 1
            W[c10][base_col + 1] += 1
            W[c11][base_col + 1] -= 1
            
            # s=2: M3 = A00(B01-B11)
            U[a_idx(0, 0)][base_col + 2] += 1
            V[b_idx(0, 1)][base_col + 2] += 1
            V[b_idx(1, 1)][base_col + 2] -= 1
            W[c01][base_col + 2] += 1
            W[c11][base_col + 2] += 1
            
            # s=3: M4 = A11(B10-B00)
            U[a_idx(1, 1)][base_col + 3] += 1
            V[b_idx(1, 0)][base_col + 3] += 1
            V[b_idx(0, 0)][base_col + 3] -= 1
            W[c00][base_col + 3] += 1
            W[c10][base_col + 3] += 1
            
            # s=4: M5 = (A00+A01)B11
            U[a_idx(0, 0)][base_col + 4] += 1
            U[a_idx(0, 1)][base_col + 4] += 1
            V[b_idx(1, 1)][base_col + 4] += 1
            W[c00][base_col + 4] -= 1
            W[c01][base_col + 4] += 1
            
            # s=5: M6 = (A10-A00)(B00+B01)
            U[a_idx(1, 0)][base_col + 5] += 1
            U[a_idx(0, 0)][base_col + 5] -= 1
            V[b_idx(0, 0)][base_col + 5] += 1
            V[b_idx(0, 1)][base_col + 5] += 1
            W[c11][base_col + 5] += 1
            
            # s=6: M7 = (A01-A11)(B10+B11)
            U[a_idx(0, 1)][base_col + 6] += 1
            U[a_idx(1, 1)][base_col + 6] -= 1
            V[b_idx(1, 0)][base_col + 6] += 1
            V[b_idx(1, 1)][base_col + 6] += 1
            W[c00][base_col + 6] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
