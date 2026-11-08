"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # 3x3 Strassen algorithm
    # Pad: A is 2x4 -> 3x4 (add zero row), B is 4x6 -> 4x3 (add zero col)
    # Actually: A (n x m) = (2 x 4), pad to (3 x 4)
    #           B (m x p) = (4 x 6), pad to (4 x 3) -- wait, that doesn't work
    # Let me reconsider.
    
    # Standard approach: use 3x3 Strassen on blocks
    # A is 2x4: think of it as 3x4 with row 2 being zero
    # B is 4x6: think of it as 4x3... no.
    
    # Actually, the right way:
    # A: 2 rows, 4 cols. Pad to 3x4 (add zero row at bottom)
    # B: 4 rows, 6 cols. Pad to 4x3... no, that changes dimensions.
    
    # Let me think differently. The 3x3 Strassen multiplies 3x3 matrices.
    # We need to tile A (2x4) and B (4x6) into 3x3 blocks.
    
    # A is 2x4: we can view as blocks. If we use 3x3 Strassen:
    # - A has 2 rows, 4 cols. Pad to 3x4 (1 zero row added).
    # - B has 4 rows, 6 cols. We need 4x3... that's not right either.
    
    # Actually: A (2x4) * B (4x6) = C (2x6)
    # Pad A to 3x4 (add row of zeros), pad B to 4x3? No...
    
    # The correct padding:
    # A: 2x4 -> pad to 3x4 (add one zero row)
    # B: 4x6 -> pad to 4x3? No, B must have 4 rows to match A's 4 cols.
    # B stays 4x6. But for 3x3 Strassen, we need B to be decomposable into 3x3 blocks.
    # B is 4x6: that's not a multiple of 3 in either dimension.
    
    # Alternative: pad A to 3x6 and B to 6x3? No...
    
    # Let me use a different approach. The 3x3 Strassen works on 3x3 matrices.
    # For A (2x4) * B (4x6):
    # Pad A to 3x6 (add 1 row, 2 cols of zeros)
    # Pad B to 6x3 (add 2 rows, 3 cols of zeros)
    # Then A_padded * B_padded = C_padded (3x3)
    # The top-left 2x6 of C_padded gives us A*B.
    
    # But 3x3 Strassen gives rank 27 for the 3x3 multiplication.
    # We only need the 2x6 output, so some of the 27 terms will have zero W coefficients.
    
    # Actually, let me reconsider. The 3x3 Strassen multiplies two 3x3 matrices.
    # A_padded is 3x6, B_padded is 6x3. That's not 3x3 * 3x3.
    
    # I think the directive means:
    # Pad A (2x4) to 3x4 by adding a zero row.
    # Pad B (4x6) to 4x3... no.
    
    # Wait, re-reading: "padding the 2x4 matrix to 3x4 and the 4x6 matrix to 4x3"
    # A: 2x4 -> 3x4 (add zero row)
    # B: 4x6 -> 4x3 (keep 4 rows, reduce to 3 cols? That loses data!)
    
    # That can't be right. Let me re-read: "pad the 2x4 matrix to 3x4 and the 4x6 matrix to 4x3"
    # Maybe it means: A becomes 3x4, B becomes 4x3? But B is 4x6, you can't reduce to 4x3.
    
    # I think the intended interpretation is:
    # Use 3x3 Strassen where:
    # - A is padded from 2x4 to 3x4 (add 1 zero row)  
    # - B is padded from 4x6 to 4x3... 
    # 
    # Actually maybe: A (2x4) -> 3x4, B (4x6) -> 4x3 doesn't make sense for multiplication.
    #
    # Let me try: pad A to 3x6 (add 1 row, 2 zero cols), pad B to 6x3 (add 2 rows, 3 zero cols)
    # Then A_pad (3x6) * B_pad (6x3) = C_pad (3x3)
    # Use 3x3 Strassen (rank 27) on this.
    # The actual A*B (2x6) is embedded in C_pad.
    #
    # But this gives rank 27 for a 3x6 * 6x3 multiplication, which is more than needed.
    
    # Simpler interpretation that works:
    # A is 2x4, B is 4x6.
    # Pad A to 3x4 (add zero row at bottom).
    # For B, we need 4 rows (matches A's 4 cols) and 6 cols.
    # Pad B to 4x6 -> but for 3x3 Strassen we need 3x3 blocks.
    # 4 is not divisible by 3, 6 is divisible by 3 (6 = 2*3).
    
    # Let me try yet another approach:
    # Pad A to 3x6 (add 1 zero row, 2 zero cols)
    # Pad B to 6x3 (add 2 zero rows, 3 zero cols)
    # A_pad: 3x6, B_pad: 6x3, C_pad: 3x3
    # Apply 3x3 Strassen to get rank 27.
    # Only output entries where i<2 and k<6 are needed (all of them since C is 2x6).
    # In C_pad (3x3), we need entries (i,k) where i in {0,1} and k in {0,1,2} (since 6/3=2... wait)
    
    # I'm overcomplicating this. Let me just use the 3x3 Strassen directly:
    # The 3x3 Strassen has 27 terms.
    # For multiplying 3x3 * 3x3, each term contributes to specific output entries.
    # 
    # For our case: A (2x4) * B (4x6)
    # Pad A to 3x4 (row 2 = 0)
    # B is 4x6. We can't directly use 3x3 Strassen unless we also pad B.
    # 
    # Actually, I think the cleanest approach:
    # Pad A from 2x4 to 3x4 (add zero row)
    # Pad B from 4x6 to 4x3... NO.
    #
    # Let me just implement 3x3 Strassen properly:
    # 3x3 Strassen multiplies X (3x3) and Y (3x3) to get Z (3x3) with 27 scalar multiplications.
    #
    # For our problem: A (2x4) * B (4x6) = C (2x6)
    # We can pad: A' = [[A, 0_{2x2}], [0_{1x4}, 0_{1x2}]] -> 3x6
    #             B' = [[B], [0_{2x6}]] -> 6x6... no that's too big.
    #
    # OK let me just go with: pad A to 3x4, and for B, note that 4x6 can be seen as
    # having 4 rows (matching A's 4 cols) and 6 cols.
    # 
    # The 3x3 Strassen applies to 3x3 matrices. To use it here:
    # - Pad A (2x4) to 3x4 by adding a zero row.
    # - For B (4x6), we need it to be compatible. 4x6 is NOT a 3x3 matrix.
    #
    # I think the correct interpretation is:
    # Use the 3x3 Strassen as a base case, and tile the problem.
    # But 2x4 and 4x6 don't tile nicely into 3x3 blocks.
    #
    # Alternative: just pad everything to 3x3:
    # A (2x4) -> pad to 3x3? No, 4 > 3.
    #
    # I think the directive's suggestion may be flawed, but let me try:
    # Pad A to 3x4 (3 rows, 4 cols)
    # Pad B to 4x3 (4 rows, 3 cols) -- but B is 4x6, so this loses 3 columns!
    #
    # Unless... we do 2 separate 3x3 Strassen calls?
    # B (4x6) = B1 (4x3) + B2 (4x3) -- split into two 4x3 halves
    # A_pad (3x4) * B1 (4x3) = C1 (3x3)
    # A_pad (3x4) * B2 (4x3) = C2 (3x3)
    # C = [C1, C2] top 2 rows = (2x6)
    # Total rank: 2 * 27 = 54. That's worse than 42!
    #
    # Hmm. Let me reconsider. Maybe pad A to 3x6 and B to 6x3:
    # A_pad (3x6) * B_pad (6x3) = C_pad (3x3)
    # 3x3 Strassen: rank 27
    # We need C (2x6) which is the top-left 2x6 of C_pad... but C_pad is only 3x3.
    # That doesn't work either.
    
    # I think the best approach within the 3x3 Strassen family:
    # Pad A (2x4) to 3x4 (add zero row)
    # For B (4x6): split into two 4x3 matrices B_left and B_right
    # Use 3x3 Strassen for A_pad * B_left (3x4 * 4x3 -> 3x3), rank 27
    # Use 3x3 Strassen for A_pad * B_right (3x4 * 4x3 -> 3x3), rank 27
    # Total rank: 54. Worse.
    
    # Or: don't split B. Instead:
    # Pad A to 3x6 (add 1 zero row, 2 zero cols)
    # Pad B to 6x3 (add 2 zero rows, 3 zero cols)
    # A_pad (3x6) * B_pad (6x3) = C_pad (3x3)
    # Use 3x3 Strassen: rank 27
    # C (2x6) is embedded in C_pad (3x3)? No, C is 2x6 but C_pad is 3x3.
    
    # I realize the dimensions don't work out for a single 3x3 Strassen.
    # Let me just keep the current 42-rank solution but verify it's correct,
    # and try to find a lower rank through optimization.
    
    # Actually, let me re-examine: the known lower bound for 2x4 x 4x6 is at least
    # 2*4*6 / something... The trivial rank is 48. Strassen block gives 42.
    # The 3x3 Strassen approach: if we could use it, rank would be 27.
    #
    # Let me try: A (2x4) pad to 3x4, B (4x6) pad to 4x3... 
    # Wait, maybe the directive means:
    # A: 2x4 -> 3x4 (pad row)
    # B: 4x6 -> 4x3 (this means we only keep first 3 cols?) No, that's lossy.
    #
    # I'll implement the current 42-rank solution correctly and add a verification step.
    # The 3x3 Strassen padding as described doesn't seem dimensionally consistent
    # for a single application. Let me keep rank 42 but ensure correctness.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen coefficients
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
