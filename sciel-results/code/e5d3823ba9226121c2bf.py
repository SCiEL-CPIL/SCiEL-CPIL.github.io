import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 4 blocks of 2x5x3
    # 2x5x3 = 2x2x3 (rank 4) + 2x3x3 (rank 6) = 10
    # Total rank = 40
    
    rank = 40
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a 1xm x px1 block? No, helper for 1xm x pxq?
    # We need to add contributions for 1x2x3 and 1x3x3 etc.
    
    # Structure:
    # A is 4x5. Split into A1 (2x5), A2 (2x5).
    # B is 5x6. Split into B1 (5x3), B2 (5x3).
    # C is 4x6. Split into C11 (2x3), C12 (2x3), C21 (2x3), C22 (2x3).
    
    # For each block (i_block, j_block) where i_block in {0,1}, j_block in {0,1}:
    # A_block is (2x5) at rows i_block*2 .. i_block*2+1
    # B_block is (5x3) at cols j_block*3 .. j_block*3+2
    # C_block is (2x3) at rows i_block*2 .. i_block*2+1, cols j_block*3 .. j_block*3+2
    
    # Inside 2x5x3:
    # Split middle 5 into 2+3.
    # A_a (2x2), A_b (2x3)
    # B_a (2x3), B_b (3x3)
    # C = A_a B_a + A_b B_b
    
    # 2x2x3 part:
    # Split A_a (2x2) into A_a1 (1x2), A_a2 (1x2)
    # B_a (2x3)
    # C_a = A_a1 B_a + A_a2 B_a
    # Each 1x2x3 has rank 2.
    
    # 2x3x3 part:
    # Split A_b (2x3) into A_b1 (1x3), A_b2 (1x3)
    # B_b (3x3)
    # C_b = A_b1 B_b + A_b2 B_b
    # Each 1x3x3 has rank 3.
    
    def add_1xm_x_pxq_block(a_row_offset, m_size, b_row_offset, p_size, c_row_offset, c_col_offset, rank_offset):
        # Adds rank m_size terms for 1xm x pxq multiplication
        # A is 1xm, B is mxq, C is 1xq
        # A index: a_row_offset (single row)
        # B index: b_row_offset .. b_row_offset+m_size-1
        # C index: c_row_offset, c_col_offset .. c_col_offset+p_size-1
        
        for k in range(m_size):
            # U
            u_idx = a_row_offset # A is 1xm, so row is fixed
            # Wait, A is 1xm. The index in U is for the whole A matrix.
            # A is n x m.
            # In our block, A is 2x5.
            # A_a1 is 1x2.
            # The row in the full A is a_row_offset.
            # The column in the full A is k.
            # So U index is a_row_offset * m + k? No.
            # U index is flat index of A.
            # A is n x m.
            # Row i, Col j. Index i*m + j.
            
            # In this helper, we are handling a sub-problem.
            # We need to map local indices to global.
            
            # This helper is too generic. Let's hardcode the loops.
            pass

    # Let's just loop through the 40 terms explicitly.
    
    # 4 Blocks (i_b, j_b)
    for i_b in range(2):
        for j_b in range(2):
            # A rows: i_b*2, i_b*2+1
            # B cols: j_b*3, j_b*3+1, j_b*3+2
            # C rows: i_b*2, i_b*2+1
            # C cols: j_b*3, j_b*3+1, j_b*3+2
            
            # Part 1: 2x2x3 (A_a 2x2, B_a 2x3)
            # A_a rows: i_b*2, i_b*2+1
            # A_a cols: 0, 1
            # B_a rows: 0, 1
            # B_a cols: j_b*3, j_b*3+1, j_b*3+2
            
            # Split A_a into 1x2 + 1x2
            # Term 1: A_a1 (1x2)
            # A_a1 row: i_b*2
            # A_a1 cols: 0, 1
            # B_a (2x3)
            # C_a1 (1x3)
            # C_a1 row: i_b*2
            # C_a1 cols: j_b*3, j_b*3+1, j_b*3+2
            
            for k in range(2): # k is 0, 1 (cols of A_a1 / rows of B_a)
                # U
                u_idx = (i_b*2) * m + k
                U[u_idx, col] = 1
                
                # V
                v_idx = k * p + (j_b*3) # Wait, B_a cols are j_b*3..
                # B_a is 2x3. Rows 0,1. Cols j_b*3..
                # V index is flat index of B.
                # B is 5x6.
                # Row k, Col j_b*3? No.
                # B_a[k, c_local] where c_local is 0..2.
                # Global col is j_b*3 + c_local.
                # But V is a vector in B-space.
                # V[v_idx, col] = 1.
                # v_idx corresponds to B[k, c_global]?
                # No, V is a vector.
                # The term is sum U * V * W.
                # U picks A[i, k].
                # V picks B[k, c].
                # W picks C[i, c].
                
                # So for a fixed k (intermediate index):
                # U should pick A[i, k].
                # V should pick B[k, c] for all c?
                # No, V is a vector.
                # If V is 1 at B[k, c], then it contributes to C[i, c].
                
                # In 1xm x pxq:
                # C[c] = sum_k A[k] B[k, c].
                # Term r=k:
                # U[k] = 1.
                # V[k, c] = 1? No.
                # V is a vector of size m*p.
                # V[k*p + c] = 1?
                # Then sum_r U[r] V[r, c] W[r, c]?
                
                # Let's check the seed.
                # U[i*m+j] = 1. (A[i,j])
                # V[j*p+k] = 1. (B[j,k])
                # W[k*n+i] = 1. (C[k,i] -> C[i,k])
                
                # So for a term corresponding to (i, j, k):
                # U[i*m+j] = 1
                # V[j*p+k] = 1
                # W[k*n+i] = 1
                
                # In our 1x2x3 case (i is fixed):
                # We sum over j (0,1).
                # For j=0:
                # U[i*m+0] = 1
                # V[0*p + c] = 1 for all c?
                # No, V is a single column in the factor matrix.
                # The factor matrix V has shape (m*p, rank).
                # V[v_idx, r] is the coefficient.
                
                # If we use rank 2 for 1x2x3:
                # r=0 (j=0):
                # U[i*m+0] = 1
                # V[0*p + c] = 1 for c in 0..2?
                # No, V[0*p + c] is a single entry.
                # We need V to pick the whole row B[0, :].
                # So V[0*p + c] = 1 for all c?
                # Then W[c*n + i] = 1 for all c?
                
                # Let's verify:
                # Sum_r U[i*m+j_r] V[j_r*p + c] W[c*n + i]
                # If r=0: j_0=0.
                # U[i*m+0] = 1.
                # V[0*p + c] = 1.
                # W[c*n + i] = 1.
                # Contribution: 1 * 1 * 1 = 1.
                # This adds to C[i, c].
                # Correct.
                
                # So for 1xm x pxq:
                # For each j in 0..m-1:
                #   r = j
                #   U[i*m+j] = 1
                #   V[j*p + c] = 1 for all c in 0..p-1
                #   W[c*n + i] = 1 for all c in 0..p-1
                
                # Wait, V and W are vectors.
                # V[j*p+c] = 1.
                # W[c*n+i] = 1.
                
                # So for our 1x2x3 block:
                # i is fixed (local 0 or 1).
                # j is 0, 1.
                # c is 0, 1, 2.
                
                # Global indices:
                # i_global = i_b*2 + i_local
                # j_global = k (0, 1)
                # c_global = j_b*3 + c_local
                
                U[i_global * m + j_global][col] = 1
                for c_local in range(3):
                    c_global = j_b * 3 + c_local
                    V[j_global * p + c_global][col] = 1
                    W[c_global * n + i_global][col] = 1
                col += 1

            # Term 2: A_a2 (1x2)
            # A_a2 row: i_b*2 + 1
            # A_a2 cols: 0, 1
            # B_a (2x3)
            # C_a2 (1x3)
            # C_a2 row: i_b*2 + 1
            # C_a2 cols: j_b*3, j_b*3+1, j_b*3+2
            
            for k in range(2):
                i_global = i_b * 2 + 1
                j_global = k
                U[i_global * m + j_global][col] = 1
                for c_local in range(3):
                    c_global = j_b * 3 + c_local
                    V[j_global * p + c_global][col] = 1
                    W[c_global * n + i_global][col] = 1
                col += 1

            # Part 2: 2x3x3 (A_b 2x3, B_b 3x3)
            # A_b rows: i_b*2, i_b*2+1
            # A_b cols: 2, 3, 4
            # B_b rows: 2, 3, 4
            # B_b cols: j_b*3, j_b*3+1, j_b*3+2
            
            # Split A_b into 1x3 + 1x3
            # Term 3: A_b1 (1x3)
            # A_b1 row: i_b*2
            # A_b1 cols: 2, 3, 4
            # B_b (3x3)
            # C_b1 (1x3)
            
            for k in range(3): # k is 0,1,2 (local index in A_b/B_b)
                j_global = 2 + k
                i_global = i_b * 2
                U[i_global * m + j_global][col] = 1
                for c_local in range(3):
                    c_global = j_b * 3 + c_local
                    V[j_global * p + c_global][col] = 1
                    W[c_global * n + i_global][col] = 1
                col += 1

            # Term 4: A_b2 (1x3)
            # A_b2 row: i_b*2 + 1
            # A_b2 cols: 2, 3, 4
            # B_b (3x3)
            # C_b2 (1x3)
            
            for k in range(3):
                j_global = 2 + k
                i_global = i_b * 2 + 1
                U[i_global * m + j_global][col] = 1
                for c_local in range(3):
                    c_global = j_b * 3 + c_local
                    V[j_global * p + c_global][col] = 1
                    W[c_global * n + i_global][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
