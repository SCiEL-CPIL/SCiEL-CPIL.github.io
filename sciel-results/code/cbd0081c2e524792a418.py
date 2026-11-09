"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    n, m, p = 6, 6, 6
    
    # Strategy: Use a 4+2 block decomposition.
    # Split matrices into 4x4, 4x2, 2x4, 2x2 blocks.
    # Use Strassen (rank 7) for 2x2 multiplications.
    # Use "Tensor Product" or direct Kronecker-like expansion for rectangular blocks?
    # Actually, for rectangular blocks like 4x2 * 2x4, the rank is 32 (schoolbook) or potentially less?
    # The rank of the tensor for m x n * n x p is at most m*n*p.
    # For 4x2 * 2x4, m=4, n=2, p=4. Rank <= 32.
    # Can we do better?
    # The tensor is T = sum_{i,j,k} e_i^A e_j^C e_k^B * delta_{...}
    # Actually, 4x2 * 2x4 is equivalent to computing 4*4 entries, each sum of 2 products.
    # C_{ik} = sum_{j=0}^1 A_{ij} B_{jk}.
    # This is just 16 independent "2-term" sums? No, A_{ij} and B_{jk} are shared.
    # However, since the middle dimension is small (2), we can't use Strassen directly on the whole block.
    # But we can view 4x2 * 2x4 as two 4x1 * 1x4 multiplications added together?
    # Let A = [A1 | A2] (4x2), B = [B1; B2] (2x4).
    # AB = A1 B1 + A2 B2.
    # A1 is 4x1, B1 is 1x4. Product is 4x4.
    # The rank of 4x1 * 1x4 is 4*1*4 = 16 (schoolbook).
    # So AB = (4x1*1x4) + (4x1*1x4).
    # If we compute these separately, rank is 16 + 16 = 32.
    # Is there a way to combine them?
    # The inputs A1, A2, B1, B2 are independent.
    # The outputs are in the same space.
    # It is unlikely we can beat 32 easily without a specific algorithm for rectangular matrices.
    # Let's assume rank 32 for 4x2*2x4.
    
    # Similarly:
    # 4x4 * 4x2:
    # A is 4x4, B is 4x2.
    # Split B into two 4x1 columns B1, B2.
    # AB = A B1 + A B2.
    # A B1 is 4x4 * 4x1. Rank 16.
    # A B2 is 4x4 * 4x1. Rank 16.
    # Total 32.
    
    # 4x2 * 2x2:
    # A is 4x2, B is 2x2.
    # Split A into two 4x1 columns A1, A2.
    # AB = A1 B1 + A2 B2.
    # A1 is 4x1, B1 is 1x2. Product 4x2. Rank 8.
    # Total 16.
    
    # 2x4 * 4x4:
    # A is 2x4, B is 4x4.
    # Split B into two 4x2 columns? Or use Strassen?
    # Let's stick to simple block splitting for rectangular to ensure correctness and simplicity.
    # A (2x4) * B (4x4).
    # Split A into two 1x4 rows? No, A is 2 rows.
    # Split B into two 4x2 blocks B1, B2.
    # AB = A B1 + A B2.
    # A (2x4) * B1 (4x2). Rank 16.
    # Total 32.
    
    # 2x4 * 4x2:
    # A (2x4) * B (4x2).
    # Rank 16.
    
    # 2x2 * 2x4:
    # A (2x2) * B (2x4).
    # Split B into two 2x2 blocks?
    # A (2x2) * B1 (2x2). Rank 7 (Strassen).
    # A (2x2) * B2 (2x2). Rank 7 (Strassen).
    # Total 14.
    
    # 2x2 * 2x2:
    # Rank 7 (Strassen).
    
    # Now sum up the ranks for the 6x6 * 6x6 multiplication using 4+2 split:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C11 = A11 B11 + A12 B21
    #   A11 B11: 4x4 * 4x4.
    #     Can we use Strassen for 4x4? Rank 49.
    #     Or just schoolbook 64?
    #     Let's use Strassen 4x4 (rank 49) for the main square blocks.
    #   A12 B21: 4x2 * 2x4. Rank 32.
    #   Total C11: 49 + 32 = 81.
    
    # C12 = A11 B12 + A12 B22
    #   A11 B12: 4x4 * 4x2. Rank 32.
    #   A12 B22: 4x2 * 2x2. Rank 16.
    #   Total C12: 32 + 16 = 48.
    
    # C21 = A21 B11 + A22 B21
    #   A21 B11: 2x4 * 4x4. Rank 32.
    #   A22 B21: 2x2 * 2x4. Rank 14.
    #   Total C21: 32 + 14 = 46.
    
    # C22 = A21 B12 + A22 B22
    #   A21 B12: 2x4 * 4x2. Rank 16.
    #   A22 B22: 2x2 * 2x2. Rank 7.
    #   Total C22: 16 + 7 = 23.
    
    # Total Rank = 81 + 48 + 46 + 23 = 198.
    
    # This is an improvement over 216 (schoolbook) and 189 (3x3 Strassen tiling)?
    # Wait, 189 is better than 198.
    # The inspiration 1 used 3x3 tiling of 2x2 Strassen blocks.
    # Rank = 9 * 7 = 63? No, that was wrong in the comment.
    # Let's re-evaluate 3x3 tiling of 2x2 Strassen.
    # We have 3x3 = 9 blocks of size 2x2.
    # C_{ij} = sum_k A_{ik} B_{kj}.
    # Each term A_{ik} B_{kj} is a 2x2 * 2x2 multiplication.
    # Rank 7 each.
    # There are 3 sums for each output block.
    # So for each output block (9 of them), we sum 3 products (3 of 7 rank = 21).
    # Total rank = 9 * 21 = 189.
    
    # So 189 is better than 198.
    # Can we do better than 189?
    # Maybe a 3+3 split?
    # 3x3 * 3x3.
    # Rank of 3x3 * 3x3 is 23 (Landsberg).
    # If we use 3+3 split:
    # C11 = A11 B11 + A12 B21.
    # A11 B11: 3x3 * 3x3. Rank 23.
    # A12 B21: 3x3 * 3x3. Rank 23.
    # Total C11: 46.
    # Similarly C12, C21, C22.
    # Total Rank = 4 * 46 = 184.
    
    # 184 < 189.
    # So the 3+3 split with rank-23 3x3 multiplication is better.
    
    # We need a rank-23 decomposition for 3x3 matrix multiplication.
    # This is the Landsberg decomposition.
    # It is complex to hardcode.
    # However, we can try to construct it or find a simpler one.
    # Is there a known simple rank-23 decomposition?
    # Yes, it exists but is not as simple as Strassen.
    
    # Given the time limit and complexity, implementing Landsberg correctly is risky.
    # Let's stick to the 189 rank decomposition (3x3 tiling of Strassen) which is easy to generate.
    # Or maybe we can improve the 4+2 split?
    # If we use Strassen for 4x4 (rank 49) in the 4+2 split, we got 198.
    # If we use schoolbook for 4x4 (rank 64), we get 64+32 + 32+16 + 32+14 + 16+7 = 213. Worse.
    
    # What if we use a 2+2+2 split?
    # That's the 3x3 tiling of 2x2. Rank 189.
    
    # What if we mix?
    # Use Strassen for the whole 6x6?
    # Strassen is recursive.
    # Rank(6) <= 7 * Rank(3)? No, Strassen splits into 4 blocks of size 3.
    # M1 = (A11+A22)(B11+B22) -> 3x3 * 3x3.
    # So Rank(6) <= 7 * Rank(3).
    # If Rank(3) = 23, then Rank(6) <= 7 * 23 = 161.
    # This is much better!
    # But we need Rank(3) decomposition.
    
    # Can we get Rank(3) = 23 easily?
    # No, it's a specific tensor.
    
    # Alternative: Use the 189 rank solution. It's safe and known.
    # Let's implement the 189 rank solution (3x3 tiling of Strassen 2x2).
    
    # Implementation of 189 rank solution:
    # U, V, W shapes: (36, 189).
    # Loop over 3x3 blocks of output (bi, bk) and sum over 3 blocks of middle (bj).
    # For each (bi, bj, bk), we do a 2x2 Strassen multiplication.
    # This contributes 7 rank-1 terms.
    # Total 3*3*3*7 = 189 terms.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen data
    strassen_A = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_B = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((0,1), 1), ((1,1), 1)],
        [((0,0), 1), ((1,0), 1)],
        [((0,0), -1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1)],
    ]
    
    col = 0
    for bi in range(3):       # Row block of A/C
        for bj in range(3):   # Col block of A / Row block of B
            for bk in range(3):  # Col block of B / Col block of C
                for t in range(7):
                    # A coefficients
                    for (ai, ak), coeff in strassen_A[t].items():
                        global_i = 2 * bi + ai
                        global_k = 2 * bj + ak
                        U[global_i * m + global_k][col] = coeff
                    
                    # B coefficients
                    for (bk_local, bj_local), coeff in strassen_B[t].items():
                        global_bk = 2 * bj + bk_local
                        global_bj = 2 * bk + bj_local
                        V[global_bk * p + global_bj][col] = coeff
                    
                    # C coefficients
                    for (ci, cj), coeff in strassen_W[t]:
                        global_i = 2 * bi + ci
                        global_j = 2 * bk + cj
                        # W index: j*n + i
                        W[global_j * n + global_i][col] = coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
