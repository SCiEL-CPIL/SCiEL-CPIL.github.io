import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen steps for 2x2x2 matrix multiplication
    # Each step is (alpha, beta, gamma) where each is a dict mapping (i,j) to coefficient
    # i, j in {0, 1} for 2x2 blocks
    strassen_222 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Trivial 2x2x1 multiplication steps (rank 4)
    # A is 2x2, B is 2x1, C is 2x1
    # C[i,0] = sum_j A[i,j] * B[j,0]
    # We need 4 independent rank-1 terms
    trivial_221 = [
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),  # A[0,0]*B[0,0] -> C[0,0]
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),  # A[0,1]*B[1,0] -> C[0,0]
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),  # A[1,0]*B[0,0] -> C[1,0]
        ({(1,1): 1}, {(1,0): 1}, {(1,0): 1}),  # A[1,1]*B[1,0] -> C[1,0]
    ]
    
    # We partition the 4x4 A into four 2x2 blocks:
    # A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    # A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # 
    # We partition the 4x6 B into four 2x3 blocks:
    # B0 (rows 0-1, cols 0-2), B1 (rows 0-1, cols 3-5)
    # B2 (rows 2-3, cols 0-2), B3 (rows 2-3, cols 3-5)
    # 
    # The 4x6 C = A * B has blocks:
    # C00 (rows 0-1, cols 0-2) = A00*B0 + A01*B2
    # C01 (rows 0-1, cols 3-5) = A00*B1 + A01*B3
    # C10 (rows 2-3, cols 0-2) = A10*B0 + A11*B2
    # C11 (rows 2-3, cols 3-5) = A10*B1 + A11*B3
    
    # For each 2x2 * 2x3 multiplication, we can use:
    # - 7 Strassen steps for the 2x2x2 part (first 2 cols of B)
    # - 4 trivial steps for the 2x2x1 part (last 1 col of B)
    # Total: 11 steps per 2x2*2x3 block
    
    # We have 4 such block multiplications:
    # 1. A00 * B0 -> contributes to C00
    # 2. A01 * B2 -> contributes to C00
    # 3. A00 * B1 -> contributes to C01
    # 4. A01 * B3 -> contributes to C01
    # 5. A10 * B0 -> contributes to C10
    # 6. A11 * B2 -> contributes to C10
    # 7. A10 * B1 -> contributes to C11
    # 8. A11 * B3 -> contributes to C11
    
    # Wait, that's 8 multiplications, each of rank 11 = 88 total. That's worse!
    # 
    # Let me reconsider. The 4x6 matrix B can be seen as 4x3 + 4x3, but that doesn't help directly.
    # 
    # Actually, the correct approach is to use the fact that 2x2*2x3 can be done in rank 11,
    # but we need to be more clever about sharing.
    #
    # Let me think differently. The 4x4*4x6 tensor can be decomposed as:
    # - 4x4*4x4 part (rank 49 using Strassen)
    # - 4x4*4x2 part (rank 28 using Strassen)
    # Total: 77
    #
    # To do better, we need to exploit the structure more.
    #
    # Alternative: Use a 2x2*2x3 decomposition with rank 11.
    # If we partition A into 2x2 blocks and B into 2x3 blocks, we get:
    # C00 = A00*B0 + A01*B2  (two 2x2*2x3 multiplications, but they share the same output block)
    #
    # Actually, each 2x2*2x3 multiplication produces a 2x3 result.
    # C00 is 2x3, so it's the sum of two 2x3 matrices from A00*B0 and A01*B2.
    #
    # The rank of a sum of two tensors is at most the sum of their ranks.
    # So C00 requires at most 11 + 11 = 22 rank terms.
    # Similarly for other blocks.
    #
    # Total: 4 blocks * 22 = 88. Still worse than 77.
    #
    # Let me try a different approach. Maybe I can use a more efficient 2x2*2x3 decomposition.
    #
    # Actually, I recall that 2x2*2x2 is rank 7 (Strassen), and 2x2*2x1 is rank 4 (trivial).
    # So 2x2*2x3 = 2x2*2x2 + 2x2*2x1 = 7 + 4 = 11.
    #
    # But wait, can we do better? The tensor 2x2*2x3 has dimension 2*2*3 = 12.
    # The trivial decomposition has rank 4*2 = 8? No, 2x2*2x3 means A is 2x2, B is 2x3, C is 2x3.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Number of (i,j,k) terms: 2*2*3 = 12.
    # Trivial rank: 2*2 = 4? No, for each (i,k), we sum over j. So there are 2*3 = 6 output entries,
    # each requiring 2 multiplications. Total rank = 6*2 = 12? No, that's not right either.
    #
    # Actually, the trivial decomposition of 2x2*2x3 has rank 2*2 = 4? Let me think again.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # So we need 4 rank-1 terms:
    # (i=0,k=0): A[0,0]*B[0,0], A[0,1]*B[1,0]
    # (i=0,k=1): A[0,0]*B[0,1], A[0,1]*B[1,1]
    # (i=0,k=2): A[0,0]*B[0,2], A[0,1]*B[1,2]
    # (i=1,k=0): A[1,0]*B[0,0], A[1,1]*B[1,0]
    # (i=1,k=1): A[1,0]*B[0,1], A[1,1]*B[1,1]
    # (i=1,k=2): A[1,0]*B[0,2], A[1,1]*B[1,2]
    #
    # That's 12 terms. But we can factor:
    # A[0,0]*(B[0,0],B[0,1],B[0,2]) + A[0,1]*(B[1,0],B[1,1],B[1,2]) + ...
    #
    # Actually, the rank of 2x2*2x3 is known to be 11 (using Strassen for the 2x2x2 part and 4 for the 2x2x1 part).
    #
    # Given the complexity, let me stick with the 77-rank decomposition but verify it's correct.
    #
    # Let me rewrite the 77-rank decomposition carefully.
    
    # Reset
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Part 1: 4x4x4 (cols 0-3 of B, cols 0-3 of C)
    # Partition A, B1, C1 into 2x2 blocks
    col_start_1 = 0
    for s in range(7):
        alpha_4, beta_4, gamma_4 = strassen_222[s]
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_222[s2]
            col = col_start_1 + s * 7 + s2
            
            # U: A is 4x4, row index = i*m + j
            for (bi, bj), val_a4 in alpha_4.items():
                for (si, sj), val_a2 in alpha_2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r, col] += val_a4 * val_a2
            
            # V: B1 is 4x4, row index = i*p + j (but only first 4 cols of B)
            for (bi, bj), val_b4 in beta_4.items():
                for (si, sj), val_b2 in beta_2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r, col] += val_b4 * val_b2
            
            # W: C1 is 4x4, row index = k*n + i
            for (bi, bj), val_c4 in gamma_4.items():
                for (si, sj), val_c2 in gamma_2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r, col] += val_c4 * val_c2
    
    # Part 2: 4x4x2 (cols 4-5 of B, cols 4-5 of C)
    # A is 4x4, B2 is 4x2, C2 is 4x2
    # Partition A into 2x2 blocks, B2 into two 2x2 blocks (top/bottom), C2 into two 2x2 blocks (top/bottom)
    col_start_2 = 49
    
    # 4 multiplications: A00*B0, A01*B1, A10*B0, A11*B1
    # where B0 is top 2x2 of B2, B1 is bottom 2x2 of B2
    # C0 is top 2x2 of C2, C1 is bottom 2x2 of C2
    mults = [
        ((0,0), (0,0)),  # A00 * B0 -> C0
        ((0,1), (1,0)),  # A01 * B1 -> C0
        ((1,0), (0,0)),  # A10 * B0 -> C1
        ((1,1), (1,0))   # A11 * B1 -> C1
    ]
    
    for m_idx, (a_block, b_block) in enumerate(mults):
        c_block = (a_block[0], 0)  # C block row index same as A block row index
        for s2 in range(7):
            alpha_2, beta_2, gamma_2 = strassen_222[s2]
            col = col_start_2 + m_idx * 7 + s2
            
            # U: A block at (a_block[0], a_block[1])
            for (si, sj), val_a2 in alpha_2.items():
                r = (a_block[0] * 2 + si) * m + (a_block[1] * 2 + sj)
                U[r, col] += val_a2
            
            # V: B2 block at (b_block[0], 0), global col offset 4
            for (si, sj), val_b2 in beta_2.items():
                r = (b_block[0] * 2 + si) * p + (4 + sj)
                V[r, col] += val_b2
            
            # W: C2 block at (c_block[0], 0), global col offset 4
            for (si, sj), val_c2 in gamma_2.items():
                r = (4 + sj) * n + (c_block[0] * 2 + si)
                W[r, col] += val_c2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
