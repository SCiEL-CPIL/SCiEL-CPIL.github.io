"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Block-level Strassen: treat 2x4 * 4x8 as 2x2 multiplication of 1x2 (U) and 2x4 (V) blocks
    # C (2x8) = [U1, U2] @ [V1; V2] where U1,U2 are 1x2, V1,V2 are 2x4
    # Split V horizontally: V1 = [V1_L(2x2), V1_R(2x2)], V2 = [V2_L(2x2), V2_R(2x2)]
    # C = [U1,V2] @ [[V1_L,V1_R],[V2_L,V2_R]]
    # C[0,:] = U1*V1_L + U1*V1_R + U2*V2_L + U2*V2_R
    # We can group: C_L = U1*V1_L + U2*V2_L (both 1x2 * 2x2 -> 1x2)
    #               C_R = U1*V1_R + U2*V2_R (both 1x2 * 2x2 -> 1x2)
    # For each of C_L and C_R, use Strassen on the 1x2 "matrices" treated as 2x2 with padding.
    # Actually, let's use a cleaner approach:
    # 
    # The 2x4 * 4x8 product C (2x8):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Block decomposition using Strassen on the 2x2 block structure:
    # Let A = [A11 A12] (2x2, 2x2) and B = [[B11] [B21]] where B11 is 2x8, B21 is 2x8
    # Wait, that doesn't give us the right shape. Let me reconsider.
    #
    # Better approach: We have 8 independent 2x2x2 multiplications (one for each pair of 
    # output columns), each using Strassen rank 7. That's 8*7=56, which is what we have.
    #
    # To potentially reduce rank, we need to exploit shared structure across the 8 blocks.
    # The key insight: the A side is only 2x4, so for different k values, the same A rows
    # are multiplied with different B rows. We can't easily share Strassen factors.
    #
    # Alternative: Use a rank-3 decomposition for 1x2x2 multiplication (since 1x2*2x2 = 1x2,
    # which is just a vector-matrix product with a 1x2 vector and 2x2 matrix -> 1x2 result)
    # Actually 1x2 @ 2x2 = 1x2, which has 2*2*1 = 4 entries. The rank is at most 4 (schoolbook).
    # There's no known sub-rank-4 decomposition for this.
    #
    # Let's try a different block structure. Consider the 2x4 * 4x8 as follows:
    # Split into 4 blocks of 2x2 * 2x2 = 2x2, where:
    #   Block 0: A[:,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    #   Block 1: A[:,0:2] @ B[0:2,2:4] -> C[0:2,2:4]
    #   Block 2: A[:,0:2] @ B[0:2,4:6] -> C[0:2,4:6]
    #   Block 3: A[:,0:2] @ B[0:2,6:8] -> C[0:2,6:8]
    #   Block 4: A[:,2:4] @ B[2:4,0:2] -> C[0:2,0:2]
    #   Block 5: A[:,2:4] @ B[2:4,2:4] -> C[0:2,2:4]
    #   Block 6: A[:,2:4] @ B[2:4,4:6] -> C[0:2,4:6]
    #   Block 7: A[:,2:4] @ B[2:4,6:8] -> C[0:2,6:8]
    #
    # This is the same 8 independent 2x2x2 blocks. Rank = 8*7 = 56.
    #
    # The theoretical lower bound for 2x4x8 is not well-known but 56 is competitive.
    # Let's verify our current decomposition is correct and just clean it up.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 16 x 56

    # Strassen 2x2x2 products
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6

    strassen = [
        # (U_coeffs for A indices, V_coeffs for B indices, W_coeffs for C indices)
        # Each is a dict from (row, col) to coefficient
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),       # P1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),                  # P2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),                  # P3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),                  # P4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),                  # P5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),                  # P6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),                  # P7
    ]

    def add_block(a_col, b_row, b_col, start_col):
        c = start_col
        for uc, vc, wc in strassen:
            for (ri, cj), coef in uc.items():
                gi = ri
                gj = a_col + cj
                U[gi * m + gj][c] = coef
            for (rj, ck), coef in vc.items():
                gj = b_row + rj
                gk = b_col + ck
                V[gj * p + gk][c] = coef
            for (rk, ri2), coef in wc.items():
                gk = b_col + rk
                gi = ri2
                W[gk * n + gi][c] = coef
            c += 1
        return c

    # 8 blocks: a_col in {0,2}, b_row in {0,2}, b_col in {0,2,4,6}
    col = 0
    for b_col in [0, 2, 4, 6]:
        col = add_block(0, 0, b_col, col)
        col = add_block(2, 2, b_col, col)

    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
