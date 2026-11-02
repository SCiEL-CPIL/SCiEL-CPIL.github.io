import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Fallback: Schoolbook decomposition (Rank 64)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Verification function
    def is_valid(U, V, W, rank):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    # Strategy: Use a 4x4 Strassen-based decomposition
    # Embed 2x4 * 4x8 into a larger structure
    # 
    # Key idea: Split the middle dimension (4) into two 2-dim halves.
    # A = [A_L | A_R] where A_L, A_R are 2x2
    # B = [B_top; B_bot] where B_top, B_bot are 2x8
    # C = A_L * B_top + A_R * B_bot  (2x8 result)
    #
    # For each 2x2 * 2x8 product, split B into four 2x2 blocks:
    # B_top = [BT0 | BT1 | BT2 | BT3]
    # B_bot = [BB0 | BB1 | BB2 | BB3]
    #
    # C = A_L*BT0 + A_L*BT1 + A_L*BT2 + A_L*BT3 + A_R*BB0 + A_R*BB1 + A_R*BB2 + A_R*BB3
    #
    # Group terms: (A_L*BT0 + A_R*BB0) + (A_L*BT1 + A_R*BB1) + ...
    # Each group is a 2x2 * 2x2 + 2x2 * 2x2 = 2x2 result
    #
    # For each group g, define:
    #   X_g = [BT_g; BB_g] (4x2)
    #   We need A_L * BT_g + A_R * BB_g
    #
    # This is equivalent to: [1, 1] * (A_L * BT_g + A_R * BB_g)
    # But we can't directly combine them into a single matrix multiply.
    #
    # Alternative: Use Strassen on the 2x2*2x2 products separately.
    # That gives 8 * 7 = 56, which is what the current code does.
    #
    # NEW APPROACH: Try to find a rank-49 decomposition by using
    # a different block structure.
    
    # Let me try: rank-49 using a 7x7 Strassen-like structure on 4x4x4
    # and extract the relevant 2x4x4x8 part.
    
    # Actually, let me try a direct approach: search for a rank-49 decomposition
    # using the known fact that m(2,4,8) <= 49.
    
    # Known result: m(2,4,8) = 49 (from Bini et al.)
    # Let me construct it using a specific algorithm.
    
    # Approach: Use the 4x4 Strassen algorithm (rank 49 for 4x4*4x4)
    # and extract the 2x4 * 4x8 subproblem.
    
    # 4x4 Strassen: M = A*B where A, B are 4x4
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    #
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B21)
    # M7 = (A12-A22)(B21+B22)
    #
    # M11 = M1+M4-M5+M7
    # M12 = M3+M5
    # M21 = M2+M4
    # M22 = M1-M2+M3+M6
    #
    # Each Mi is a 2x2*2x2 product (rank 7 using Strassen)
    # Total rank for 4x4*4x4 = 7*7 = 49
    #
    # Now, our problem: A(2x4) * B(4x8)
    # Embed A into a 4x4 matrix: A_full = [[A, 0], [0, 0]] where A is 2x4
    #   A_full = [[A11, A12, 0, 0],
    #             [A21, A22, 0, 0],
    #             [0, 0, 0, 0],
    #             [0, 0, 0, 0]]
    #   where A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    #   But A is only 2x4, so A21 = A22 = 0.
    #
    # Embed B into a 4x8 matrix... but Strassen needs square matrices.
    # This approach doesn't directly work for rectangular.
    
    # Let me try a different embedding.
    
    # Alternative: Use the 4x4 Strassen structure but adapted for 2x4 * 4x8.
    
    # Split the 4x8 B matrix into 2x2 blocks:
    # B = [[B11, B12, B13, B14],
    #      [B21, B22, B23, B24],
    #      [B31, B32, B33, B34],
    #      [B41, B42, B43, B44]]
    # where each Bij is 1x2 (a row of B).
    
    # This is getting complicated. Let me try a simpler structural change.
    
    # NEW STRATEGY: Use a 2x4 * 4x4 multiplication with rank 14,
    # applied to two 4x4 halves of B.
    
    # For 2x4 * 4x4: 
    # A = [A1 | A2] (2x2 each)
    # B = [[B1, B2], [B3, B4]] (2x2 each)
    # C = A1*B1 + A2*B3 (2x2) and A1*B2 + A2*B4 (2x2)
    # 
    # Using Strassen on each 2x2*2x2: rank 7 each
    # Total: 4 * 7 = 28 for 2x4 * 4x4
    # For 2x4 * 4x8: 2 * 28 = 56. Same as before.
    
    # Let me try to find a rank-49 decomposition by sharing computations.
    
    # Key insight: In the 4x4 Strassen, the 7 multiplications are of 2x2 matrices.
    # If we can map our 2x4 * 4x8 problem into a subset of the 4x4 * 4x4 problem,
    # we might get rank 49.
    
    # Let me try: treat the problem as computing C = A*B where A is 2x4, B is 4x8.
    # Pad A to 4x4: A_pad = [[A, 0_2x4], [0_2x4, 0_2x4]]
    # Pad B to 4x8: B_pad = [[B, 0_4x4], [0_4x4, 0_4x4]]... no, B is already 4x8.
    
    # This padding approach is messy. Let me try a direct rank-49 construction.
    
    # Known: m(2,4,8) = 49. Let me use a specific algorithm.
    
    # From the literature, one way to get rank 49 for 2x4 * 4x8:
    # Use the 4x4 Strassen (rank 49) and note that our problem is a subproblem.
    
    # Actually, let me just try to verify if rank 56 is optimal or if we can do better.
    # For now, let me stick with rank 56 but try a different implementation
    # that might be more efficient or allow for future improvements.
    
    # Let me try a rank-49 construction using a different method.
    
    # Method: Use 7 groups of 7 bilinear forms each.
    # Total: 49 bilinear forms.
    
    # I'll use a heuristic search to find a rank-49 decomposition.
    
    def try_rank49():
        """Try to find a rank-49 decomposition using a structured approach."""
        rank = 49
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Use the 4x4 Strassen structure adapted for 2x4 * 4x8
        # 
        # Let me define the 7 Strassen-like combinations:
        # 
        # For the 2x4 * 4x8 problem, I'll use a block structure where
        # I split the 4 middle dimension into two 2-dim halves, and
        # the 8 output dimension into four 2-dim quarters.
        #
        # A = [A_L | A_R] (2x2 each)
        # B = [B_0 | B_1 | B_2 | B_3] (2x2 each, stacked as 4x8)
        # C = [C_0 | C_1 | C_2 | C_3] (2x2 each)
        #
        # C_g = A_L * B_g_top + A_R * B_g_bot
        #
        # Where B_g = [B_g_top; B_g_bot] (4x2)
        #
        # Now, for each g, I need to compute A_L * B_g_top + A_R * B_g_bot.
        # This is a 2x2 * 2x2 + 2x2 * 2x2 = 2x2 product.
        #
        # I can write this as: [A_L, A_R] * [B_g_top; B_g_bot] but that's 2x4 * 4x2.
        #
        # Using Strassen on the 2x4 * 4x2 product:
        # Split [A_L, A_R] into 2x2 blocks: AL, AR
        # Split [B_g_top; B_g_bot] into 2x2 blocks: BT, BB
        #
        # C_g = AL*BT + AR*BB
        #
        # This is exactly a 2x2 * 2x2 + 2x2 * 2x2 = 2x2 problem.
        #
        # The tensor for 2x2*2x2 + 2x2*2x2 = 2x2 has rank at most 14 (two Strassens).
        # But can we do better by combining?
        #
        # Let me try: Use 7 bilinear forms to compute AL*BT + AR*BB.
        #
        # Define:
        # X1 = AL + AR
        # Y1 = BT + BB
        # M1 = X1 * Y1 = AL*BT + AL*BB + AR*BT + AR*BB
        #
        # X2 = AR
        # Y2 = BT
        # M2 = X2 * Y2 = AR*BT
        #
        # X3 = AL
        # Y3 = BB
        # M3 = X3 * Y3 = AL*BB
        #
        # Then AL*BT + AR*BB = M1 - M2 - M3
        #
        # But M1, M2, M3 are each 2x2*2x2 products, each needing 7 bilinear forms.
        # Total: 21 bilinear forms per g, times 4 = 84. Worse than 56!
        
        # This approach doesn't work. Let me stick with rank 56.
        return None
    
    # For now, return the rank-56 Strassen decomposition
    def strassen_rank56():
        total_rank = 56
        U = np.zeros((nm, total_rank))
        V = np.zeros((mp, total_rank))
        W = np.zeros((pn, total_rank))
        
        def add_strassen(a_idx, b_idx, c_idx, base_col):
            a00, a01, a10, a11 = a_idx
            b00, b01, b10, b11 = b_idx
            c00, c01, c10, c11 = c_idx
            
            # M1 = (A00+A11)(B00+B11)
            U[a00, base_col] += 1
            U[a11, base_col] += 1
            V[b00, base_col] += 1
            V[b11, base_col] += 1
            W[c00, base_col] += 1
            W[c11, base_col] += 1
            
            # M2 = (A10+A11)B00
            U[a10, base_col + 1] += 1
            U[a11, base_col + 1] += 1
            V[b00, base_col + 1] += 1
            W[c10, base_col + 1] += 1
            W[c11, base_col + 1] -= 1
            
            # M3 = A00(B10-B11)
            U[a00, base_col + 2] += 1
            V[b10, base_col + 2] += 1
            V[b11, base_col + 2] -= 1
            W[c01, base_col + 2] += 1
            W[c11, base_col + 2] += 1
            
            # M4 = A11(B01-B10)
            U[a11, base_col + 3] += 1
            V[b01, base_col + 3] += 1
            V[b10, base_col + 3] -= 1
            W[c00, base_col + 3] += 1
            W[c10, base_col + 3] += 1
            
            # M5 = (A00+A01)B11
            U[a00, base_col + 4] += 1
            U[a01, base_col + 4] += 1
            V[b11, base_col + 4] += 1
            W[c00, base_col + 4] -= 1
            W[c01, base_col + 4] += 1
            
            # M6 = (A10-A00)(B00+B10)
            U[a10, base_col + 5] += 1
            U[a00, base_col + 5] -= 1
            V[b00, base_col + 5] += 1
            V[b10, base_col + 5] += 1
            W[c11, base_col + 5] += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[a01, base_col + 6] += 1
            U[a11, base_col + 6] -= 1
            V[b10, base_col + 6] += 1
            V[b11, base_col + 6] += 1
            W[c00, base_col + 6] += 1

        for g in range(4):
            a_L_idx = [0, 1, 4, 5]
            b_top_idx = [2*g, 2*g+1, 8+2*g, 8+2*g+1]
            c_g_idx = [4*g, 4*g+1, 4*g+2, 4*g+3]
            
            add_strassen(a_L_idx, b_top_idx, c_g_idx, g * 14)
            
            a_R_idx = [2, 3, 6, 7]
            b_bot_idx = [16+2*g, 16+2*g+1, 24+2*g, 24+2*g+1]
            
            add_strassen(a_R_idx, b_bot_idx, c_g_idx, g * 14 + 7)

        return U, V, W

    U56, V56, W56 = strassen_rank56()
    if is_valid(U56, V56, W56, 56):
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    else:
        Us, Vs, Ws = schoolbook()
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
# EVOLVE-BLOCK-END
