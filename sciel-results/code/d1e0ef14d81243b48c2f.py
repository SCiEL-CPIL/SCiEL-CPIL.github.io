import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook Rank 30
    # U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # We try to optimize the 2x2x2 sub-block using Strassen (Rank 7 instead of 8)
    # This saves 1 rank.
    # We try to optimize the 2x2x3 sub-block using Rank 11 (instead of 12).
    # This saves 1 rank.
    # Total Target Rank: 28.
    
    # Since I don't have the explicit Rank 11 coefficients for 2x2x3 memorized,
    # I will implement the Strassen 2x2x2 part which is well-known, and leave the rest schoolbook.
    # This guarantees Rank 29.
    
    # Structure:
    # A (2x3) = [A1 (2x2) | A2 (2x1)]
    # B (3x5) = [B1 (2x5); B2 (1x5)]
    # C = A1 B1 + A2 B2
    
    # B1 (2x5) can be split into B1a (2x2) and B1b (2x3).
    # A1 B1a is 2x2x2. Use Strassen (Rank 7).
    # A1 B1b is 2x2x3. Use Schoolbook (Rank 12).
    # A2 B2 is 2x1x5. Use Schoolbook (Rank 5).
    
    # Total Rank = 7 + 12 + 5 = 24?
    # If this decomposition is valid, the rank is 24.
    # But the knowledge says 28.
    # This implies the blocks are NOT disjoint in the tensor space or my additivity is wrong.
    # The tensor rank is NOT additive for disjoint supports in this way?
    # Actually, if the supports are disjoint in the output C, the ranks add.
    # C1 = A1 B1a (cols 0,1 of C)
    # C2 = A1 B1b (cols 2,3,4 of C)
    # C3 = A2 B2 (ALL cols of C, added to C1 and C2)
    # Ah, C3 adds to ALL columns. So it overlaps.
    # C = A1 B1 + A2 B2
    # C_{i,k} = (A1 B1)_{i,k} + (A2 B2)_{i,k}
    # So we cannot simply sum ranks.
    
    # Therefore, I will fall back to the safe Schoolbook Rank 30.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
