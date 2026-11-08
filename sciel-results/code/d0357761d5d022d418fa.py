import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    # The current approach uses a rank-98 decomposition based on 2x2 Strassen blocks.
    # Rank 98 = 2 halves * 7 Strassen steps * 7 2x2x2 components.
    # We will stick to this proven design family but ensure correctness and speed.
    
    # Initialize U, V, W
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition details
    # U_2, V_2, W_2 are 4x7
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # c=0
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # c=1
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # c=2
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # c=3
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # c=4
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # c=5
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # c=6
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1
    
    # Block Strassen coefficients for 4x4x4
    # P_k = sum_{blocks} coeff * A_block * B_block
    # C_block = sum_{P_k} coeff * P_k
    
    # We need to map block indices to global indices.
    # A is 4x4. Blocks are 2x2.
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B is 4x8 (for the full problem) or 4x4 (for the sub-problem).
    # In the sub-problem (4x4x4), B is 4x4.
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # For the full 4x4x8 problem, we split B into B_L (cols 0-3) and B_R (cols 4-7).
    # So for half L, B indices are 0-3. For half R, B indices are 4-7.
    
    # Let's define the contributions for each P_k in terms of A_block and B_block indices.
    # A_block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B_block indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    P_defs = [
        # P0
        { (0,0): 1, (3,0): 1, (0,3): 1, (3,3): 1 }, # (A00+A11)(B00+B11) -> A00B00 + A00B11 + A11B00 + A11B11
        # Wait, Strassen formula is (A00+A11)(B00+B11). This expands to 4 terms.
        # But in the tensor decomposition, we treat (A00+A11) as a single matrix A' and (B00+B11) as B'.
        # The rank 7 decomposition applies to A' * B'.
        # So for P0, we use A' = A00 + A11 and B' = B00 + B11.
        # This means for the 2x2x2 decomposition, the input A is A' and input B is B'.
        # We need to map the 2x2 indices of A' to the 4x4 indices of A.
        # A'_{ij} = A_{block(i,j)}_{ij}.
        
        # This approach is slightly tricky to implement directly with simple index mapping if we just add up contributions.
        # Instead, we can iterate through the 7 P_k's.
        # For each P_k, we determine the "effective" A_block and B_block.
        # Then we apply the 2x2x2 Strassen decomposition to that effective block.
        
        # Let's define the effective A and B for each P_k.
        # P0: A_eff = A00 + A11, B_eff = B00 + B11
        # P1: A_eff = A10 + A11, B_eff = B00
        # P2: A_eff = A00, B_eff = B01 - B11
        # P3: A_eff = A11, B_eff = B10 - B00
        # P4: A_eff = A00 + A01, B_eff = B11
        # P5: A_eff = A10 - A00, B_eff = B00 + B01
        # P6: A_eff = A01 - A11, B_eff = B10 + B11
        
        # For each P_k, we run the 2x2x2 decomposition.
        # The 2x2x2 decomposition has 7 components S_m.
        # For each S_m, we have U_2, V_2, W_2 values.
        # U_2 acts on A_eff. V_2 acts on B_eff. W_2 acts on C_eff (which is P_k).
        
        # We need to map A_eff indices (0..3) to global A indices.
        # A_eff is a 2x2 matrix.
        # A_eff[0,0] corresponds to A[0,0] if A_eff=A00, etc.
        
        # Let's define a function to get global indices for a block.
        # Block (bi, bj) of 2x2 size in a 4x4 matrix.
        # Global row: bi*2 + r, Global col: bj*2 + c.
        
        # For A_eff = A00 + A11:
        # A_eff[0,0] = A00[0,0] + A11[0,0] -> A[0,0] + A[2,2]
        # A_eff[0,1] = A00[0,1] + A11[0,1] -> A[0,1] + A[2,3]
        # A_eff[1,0] = A00[1,0] + A11[1,0] -> A[1,0] + A[3,2]
        # A_eff[1,1] = A00[1,1] + A11[1,1] -> A[1,1] + A[3,3]
        
        # So for U_2[row, m], we add U_2[row, m] to U[global_idx, col].
        
        # We will precompute the mapping for each P_k and each of the 4 indices of the 2x2 block.
    ]
    
    # This is getting complex to hardcode. Let's just generate it programmatically.
    
    # Define the 7 P_k definitions as lists of (coeff, a_block, b_block)
    # a_block, b_block are tuples (bi, bj)
    
    P_specs = [
        # P0
        [ (1, (0,0)), (1, (1,1)), (1, (0,0)), (1, (1,1)) ], # This is wrong, Strassen is (A00+A11)(B00+B11)
        # It's a single product of sums.
        # A_eff = A00 + A11
        # B_eff = B00 + B11
    ]
    
    # Let's define A_eff and B_eff specs directly.
    # Each spec is a list of (coeff, block_idx)
    # block_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    A_specs = [
        [(1, 0), (1, 3)], # P0: A00 + A11
        [(1, 2), (1, 3)], # P1: A10 + A11
        [(1, 0)],         # P2: A00
        [(1, 3)],         # P3: A11
        [(1, 0), (1, 1)], # P4: A00 + A01
        [(1, 2), (-1, 0)],# P5: A10 - A00
        [(1, 1), (-1, 3)],# P6: A01 - A11
    ]
    
    B_specs = [
        [(1, 0), (1, 3)], # P0: B00 + B11
        [(1, 0)],         # P1: B00
        [(1, 1), (-1, 3)],# P2: B01 - B11
        [(1, 2), (-1, 0)],# P3: B10 - B00
        [(1, 3)],         # P4: B11
        [(1, 0), (1, 1)], # P5: B00 + B01
        [(1, 2), (1, 3)], # P6: B10 + B11
    ]
    
    # C_specs: How P_k contributes to C blocks
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    C_specs = [
        # P0
        [(1, 0), (1, 3)], # C00 + C11
        # P1
        [(1, 2), (-1, 3)], # C10 - C11
        # P2
        [(1, 1), (1, 3)], # C01 + C11
        # P3
        [(1, 0), (1, 2)], # C00 + C10
        # P4
        [(-1, 0), (1, 1)], # -C00 + C01
        # P5
        [(1, 3)], # C11
        # P6
        [(1, 0)], # C00
    ]
    
    # Now, for each half (L, R) and each P_k (0..6) and each S_m (0..6):
    
    for half in range(2): # 0 for L, 1 for R
        # B offset for this half
        # B is 4x8.
        # L: cols 0-3. R: cols 4-7.
        # In the 4x4 sub-problem, B is 4x4.
        # Global B index: j_global * 8 + k_global.
        # For half L, k_global = k_sub.
        # For half R, k_global = k_sub + 4.
        
        for k_p in range(7): # P_k
            for m_s in range(7): # S_m
            
                col = half * 49 + k_p * 7 + m_s
                
                # --- U (Input A) ---
                # A is 4x4. No offset.
                # A_specs[k_p] gives contributions to A_eff (2x2).
                # U_2[m_s] gives coefficients for A_eff indices (0..3).
                
                # A_eff index 0: (0,0)
                # A_eff index 1: (0,1)
                # A_eff index 2: (1,0)
                # A_eff index 3: (1,1)
                
                # We need to map A_eff indices to global A indices.
                # A_eff[r, c] = sum_{(coeff, block) in A_specs[k_p]} coeff * A_block[r, c]
                # A_block[r, c] is at global A index (bi*2+r)*4 + (bj*2+c)
                
                for r in range(2):
                    for c in range(2):
                        a_eff_idx = r * 2 + c
                        coeff_u2 = U_2[a_eff_idx, m_s]
                        if coeff_u2 == 0: continue
                        
                        total_coeff = 0
                        for coeff_a, block_idx in A_specs[k_p]:
                            bi, bj = divmod(block_idx, 2)
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            total_coeff += coeff_a * coeff_u2
                            
                            # Wait, U_2 is fixed for S_m.
                            # The sum is over the blocks that form A_eff.
                            # So for a fixed a_eff_idx, we sum over blocks.
                            # But we need to add to U[global_a_idx, col].
                            
                        # Actually, it's better to iterate blocks and add contributions.
                        # Let's restart the U filling logic.
                        
                        pass # Placeholder
                
                # Correct U filling:
                # For each block in A_specs[k_p]:
                #   For each (r, c) in 2x2:
                #     global_a_idx = ...
                #     U[global_a_idx, col] += coeff_a * U_2[r*2+c, m_s]
                
                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_a_idx = global_r * 4 + global_c
                            a_eff_idx = r * 2 + c
                            U[global_a_idx, col] += coeff_a * U_2[a_eff_idx, m_s]
                
                # --- V (Input B) ---
                # B is 4x8.
                # B_specs[k_p] gives contributions to B_eff (2x2).
                # B_eff[r, c] = sum_{(coeff, block) in B_specs[k_p]} coeff * B_block[r, c]
                # B_block[r, c] is at global B index (bi*2+r)*8 + (bj*2+c + half*4)
                
                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            global_r = bi * 2 + r
                            global_c = bj * 2 + c
                            global_k = global_c + half * 4
                            global_b_idx = global_r * 8 + global_k
                            b_eff_idx = r * 2 + c
                            V[global_b_idx, col] += coeff_b * V_2[b_eff_idx, m_s]
                
                # --- W (Output C) ---
                # C is 4x8.
                # C_specs[k_p] gives contributions of P_k to C blocks.
                # C_block[bi, bj] += coeff_c * P_k
                # P_k is a 2x2 matrix (result of A_eff * B_eff).
                # W_2[m_s] gives coefficients for P_k indices (0..3).
                # P_k[r, c] contributes to C_block[r, c].
                # C_block[bi, bj] is at global C index (bk*2+r)*4 + (bi*2+c)
                # Note: W index is k*n + i.
                # C is 4x8? No, C is 4x8?
                # A(4x4) * B(4x8) = C(4x8).
                # W shape is p*n = 8*4 = 32.
                # Index k*4 + i.
                
                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    # c_block_idx is the block index of C.
                    # C has 2x2 blocks of size 2x2?
                    # C is 4x8.
                    # We split C into 2x4 blocks? Or 4x2?
                    # The Strassen block structure for C in 4x4x4 was 2x2 blocks of 2x2.
                    # Here C is 4x8.
                    # We split B into two 4x4s.
                    # So C is split into two 4x4s: C_L and C_R.
                    # C_L is 4x4. It has 2x2 blocks of 2x2.
                    # C_R is 4x4. It has 2x2 blocks of 2x2.
                    
                    # So for half L, C is C_L.
                    # For half R, C is C_R.
                    
                    # Global C index for block (bi, bj) of C_half:
                    # C_half is 4x4.
                    # Block (bi, bj) is rows bi*2..bi*2+1, cols bj*2..bj*2+1.
                    # Global C row: i_global = bi*2 + r.
                    # Global C col: k_global = bj*2 + c + half*4.
                    # W index: k_global * 4 + i_global.
                    
                    for r in range(2):
                        for c in range(2):
                            global_i = bi * 2 + r
                            global_k = bj * 2 + c + half * 4
                            global_w_idx = global_k * 4 + global_i
                            p_eff_idx = r * 2 + c
                            W[global_w_idx, col] += coeff_c * W_2[p_eff_idx, m_s]

    # Check deadline
    if time.time() - start_time > DEADLINE:
        # If we're too slow, return what we have (though this loop is fast)
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
