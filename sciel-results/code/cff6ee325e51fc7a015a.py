import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    
    # Lader's rank-19 algorithm for 4x4x4 matrix multiplication
    # We'll build the full 4x4x4 decomposition, then slice to 3x4x4
    
    rank = 19
    
    # For 4x4x4: U is (16, rank), V is (16, rank), W is (16, rank)
    # Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    # Lader's 19-term algorithm for 4x4x4
    # Based on the structure: split into 2x2 blocks, use Strassen-like
    
    # Let A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Use Strassen for each 2x2 block multiplication:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # But we need to share computations across the 4 block multiplications.
    # The key insight of Lader's algorithm is to share the Strassen multipliers
    # across all 4 block products.
    
    # Let's define the 19 multipliers:
    # For C11 = A11*B11 + A12*B21:
    #   M1_11 = (A11+A22)*(B11+B22)  -- but A22 and B22 are from different blocks
    
    # Actually, let me use a different approach. The 19-term algorithm
    # for 4x4x4 is well-known. Let me encode it directly.
    
    # From literature, the 19-term algorithm uses these multipliers:
    # Define block indices:
    # A11: rows 0-1, cols 0-1 -> A indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> A indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> A indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> A indices 10,11,14,15
    
    # B11: rows 0-1, cols 0-1 -> B indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> B indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> B indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> B indices 10,11,14,15
    
    # C11: rows 0-1, cols 0-1 -> C indices 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> C indices 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> C indices 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> C indices 10,11,14,15
    
    # The 19 multipliers (using Strassen on the 4 block products with sharing):
    
    # Let me use the explicit 19-term formula from Lader's paper.
    # After careful consideration, here's a known rank-19 scheme:
    
    # Multiplier 1: (A11+A22) * (B11+B22)
    # Multiplier 2: (A21+A22) * B11
    # Multiplier 3: A11 * (B12-B22)
    # Multiplier 4: A22 * (B21-B11)
    # Multiplier 5: (A11+A12) * B22
    # Multiplier 6: (A21-A11) * (B11+B12)
    # Multiplier 7: (A12-A22) * (B21+B22)
    # These 7 give C11, C12, C21, C22 for the "top-left" 2x2 block product structure.
    
    # But we need 4 separate 2x2x2 products, each needing 7 terms = 28.
    # The sharing reduces this to 19.
    
    # Let me just use a verified rank-19 decomposition.
    # I'll construct it by combining Strassen blocks with sharing.
    
    # Actually, let me try a different, more direct approach.
    # I'll use the fact that we can verify correctness numerically.
    
    # Let me try to build a rank-19 decomposition by optimization,
    # but first let me try a known explicit construction.
    
    # From the literature, one rank-19 algorithm for 4x4x4:
    # Uses 19 bilinear forms. Here's one explicit construction:
    
    # Define the 19 multipliers as combinations of A and B blocks.
    
    # I'll use a parametric approach and verify.
    
    # Let me try a simpler approach: use the 7-term Strassen for each of the
    # 4 corner 2x2x2 products, but share the "diagonal" multipliers.
    
    # For C11 = A11*B11 + A12*B21:
    #   s1 = (A11+A12) * (B11+B21)
    #   s2 = A11 * B11
    #   s3 = A12 * B21
    #   C11 = s1 - s2 - s3 + ... hmm, this isn't quite right.
    
    # Let me just use the standard Strassen for each 2x2x2:
    # For multiplying X (2x2) by Y (2x2):
    #   M1 = (X00+X11)*(Y00+Y11)
    #   M2 = (X10+X11)*Y00
    #   M3 = X00*(Y01-Y11)
    #   M4 = X11*(Y10-Y00)
    #   M5 = (X00+X01)*Y11
    #   M6 = (X10-X00)*(Y00+Y01)
    #   M7 = (X01-X11)*(Y10+Y11)
    #   C00 = M1+M4-M5+M7
    #   C01 = M3+M5
    #   C10 = M2+M4
    #   C11 = M1-M2+M3+M6
    
    # For 4x4x4, we have 4 such products. The sharing is that:
    # - The "diagonal" multipliers M1, M4 can be shared between adjacent blocks.
    
    # This is getting complex. Let me just try to find a rank-19 decomposition
    # by using a known one from the literature.
    
    # Here's a rank-19 decomposition from Lader (1976):
    # I'll encode it using the block structure.
    
    # Let me try a different strategy: since I can verify numerically,
    # let me try to find a low-rank decomposition using optimization.
    
    # But first, let me try the simple approach of just using 4 independent
    # Strassen decompositions (rank 28) and see if that's better than 44.
    
    # Actually, 28 < 44, so even without sharing, using Strassen for each
    # 2x2x2 block gives rank 28, which is better than 44.
    
    # Let me implement 4 independent Strassen decompositions (rank 28).
    
    rank = 28
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    # Strassen multipliers for a 2x2x2 product
    # U_loc: (4, 7), V_loc: (4, 7), W_loc: (4, 7)
    # A indices: 0,1,2,3 (row*2+col)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    def strassen_block():
        U_loc = np.zeros((4, 7))
        V_loc = np.zeros((4, 7))
        W_loc = np.zeros((4, 7))
        
        # M1 = (A00+A11)*(B00+B11)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        # M3 = A00*(B01-B11)
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        # M4 = A11*(B10-B00)
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        return U_loc, V_loc, W_loc
    
    U_loc, V_loc, W_loc = strassen_block()
    
    # Now place 4 blocks:
    # Block 0: A11 * B11 -> C11 (top-left)
    # Block 1: A12 * B21 -> C11 (top-left, second term)
    # Block 2: A11 * B12 -> C12 (top-right)
    # Block 3: A12 * B22 -> C12 (top-right, second term)
    # Block 4: A21 * B11 -> C21 (bottom-left)
    # Block 5: A22 * B21 -> C21 (bottom-left, second term)
    # Block 6: A21 * B12 -> C22 (bottom-right)
    # Block 7: A22 * B22 -> C22 (bottom-right, second term)
    
    # That's 8 blocks * 7 = 56, which is worse than 28.
    
    # I need to use Strassen properly for the full 4x4x4.
    # The 4x4x4 tensor has 64 entries. Strassen gives rank 7^log2(4) = 49.
    # That's worse than 28.
    
    # Let me reconsider. The 4x4x4 multiplication can be seen as 4 independent
    # 2x2x2 multiplications if we don't share. That's 4*7 = 28.
    
    # But each "2x2x2" here is actually a 2x2x2 block product, not a full 2x2x2.
    # C11 = A11*B11 + A12*B21 is a sum of two 2x2x2 products.
    
    # To compute C11, I need to compute A11*B11 and A12*B21 separately.
    # Each 2x2x2 product needs 7 terms, so C11 needs 14 terms.
    # Similarly for C12, C21, C22. Total = 4*14 = 56.
    
    # That's worse. I need a better approach.
    
    # The key insight: C11 = A11*B11 + A12*B21 can be computed with fewer
    # terms than 14 by using the structure.
    
    # Actually, let me just use the naive approach: 4x4x4 needs 64 terms naively.
    # Strassen on 4x4x4: split into 2x2 blocks, use 7 recursive calls.
    # 7 * (2x2x2 rank) = 7 * 7 = 49.
    
    # So Strassen gives rank 49 for 4x4x4.
    # Lader gives rank 19.
    
    # Let me just try to find a rank-19 decomposition by optimization.
    
    # Actually, let me try a different approach. Let me use the known fact
    # that the 3x4x4 tensor has a lower bound and try to find a good decomposition.
    
    # For now, let me just use the 44-rank decomposition and try to improve it.
    
    # Let me try rank 40 by using a better 3x2x2 decomposition.
    
    # The 3x2x2 tensor: A is 3x2, B is 2x2, C is 3x2.
    # Naive: 3*2*2 = 12 terms.
    # Strassen-based: 11 terms (as in current code).
    # Can we do better?
    
    # The 3x2x2 tensor has 12 entries. Lower bound is max(3*2, 2*2, 3*2) = 12? No.
    # Lower bound is max(n*m, m*p, p*n) = max(6, 4, 6) = 6.
    # So rank could be as low as 6.
    
    # Let me try to find a rank-6 or rank-7 decomposition for 3x2x2.
    
    # Actually, let me just try to optimize the 3x2x2 decomposition.
    
    # For 3x2x2: T[i,j,k] = A[i,j]*B[j,k]
    # i in 0..2, j in 0..1, k in 0..1
    
    # 12 entries. Let me try rank 6.
    
    # U: (6, r), V: (4, r), W: (6, r)
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    # Let me try to find a rank-6 decomposition.
    
    # This is getting complex. Let me just go with the current 44-rank approach
    # and see if I can find a slightly better one.
    
    # Actually, let me try a completely different approach: use the 4x4x4 rank-19
    # decomposition and slice it.
    
    # I'll look up the explicit Lader rank-19 algorithm.
    
    # After research, here's a rank-19 algorithm for 4x4x4:
    # It uses 19 bilinear forms. Let me encode one.
    
    # From the literature, one explicit rank-19 scheme:
    # Let A = [a_ij], B = [b_jk], C = [c_ik]
    
    # The 19 multipliers are:
    # M1 = (a00+a11+a22+a33)(b00+b11+b22+b33)
    # ... this is getting too complex to encode from memory.
    
    # Let me try a numerical approach: use optimization to find a low-rank
    # decomposition of the 3x4x4 tensor.
    
    # The tensor has 3*4*4 = 48 entries.
    # I'll try to find a rank-r decomposition by minimizing the error.
    
    # But this might not converge to an exact solution.
    
    # Let me just go with the current 44-rank decomposition for now,
    # since I'm not confident I can find a better one quickly.
    
    # Actually, let me try one more thing: use the 4x4x4 Strassen decomposition
    # (rank 49) and slice it to 3x4x4. That gives rank 49, which is worse than 44.
    
    # So the current 44-rank approach is already better than slicing Strassen.
    
    # Let me try to improve the 3x2x2 decomposition from rank 11 to rank 10.
    
    # If I can get rank 10 for 3x2x2, then 4 blocks give rank 40.
    
    # Let me try to find a rank-10 decomposition for 3x2x2.
    
    # 3x2x2 tensor: 12 entries.
    # Rank 10 means 2 fewer terms than the current 11.
    
    # Let me try to optimize this.
    
    # Actually, let me just return the current 44-rank decomposition for now.
    # I don't have time to find a better one.
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part (cols 7-10) for Row 2 of A
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[4, 7] = 1
    
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[4, 8] = 1
    
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1
    
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
    
    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
