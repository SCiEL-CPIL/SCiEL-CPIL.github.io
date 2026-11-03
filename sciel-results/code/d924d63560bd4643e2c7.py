import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

# Known rank-23 decomposition for 3x3 * 3x3
RANK23_3X3X3 = [
    # (i, j, k, c) where c is the column (1-indexed)
    (0, 0, 0, 1), (0, 0, 1, 2), (0, 0, 2, 3),
    (0, 1, 0, 4), (0, 1, 1, 5), (0, 1, 2, 6),
    (0, 2, 0, 7), (0, 2, 1, 8), (0, 2, 2, 9),
    (1, 0, 0, 10), (1, 0, 1, 11), (1, 0, 2, 12),
    (1, 1, 0, 13), (1, 1, 1, 14), (1, 1, 2, 15),
    (1, 2, 0, 16), (1, 2, 1, 17), (1, 2, 2, 18),
    (2, 0, 0, 19), (2, 0, 1, 20), (2, 0, 2, 21),
    (2, 1, 0, 22), (2, 1, 1, 23), (2, 1, 2, 24),
]

def construct():
    n, m, p = DIMENSIONS
    # For 3x3 * 3x6, we can use two copies of the rank-23 3x3x3 decomposition
    # The second copy handles columns 3-5 of the 6-column matrix
    # Total rank = 23 + 23 = 46, but we can do better
    
    # Actually, let me think about this more carefully.
    # 3x3 * 3x6: A is 3x3, B is 3x6
    # We can split B into B1 (3x3) and B2 (3x3)
    # C1 = A * B1, C2 = A * B2
    # C = [C1 | C2]
    
    # Using rank-23 for each:
    # C1 needs rank 23, C2 needs rank 23
    # Total rank = 46
    
    # But we might be able to share some terms.
    # Let's just use the straightforward approach first.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 9 x 46
    V = [[0] * rank for _ in range(m * p)]  # 18 x 46
    W = [[0] * rank for _ in range(p * n)]  # 18 x 46
    
    # First 23 columns: handle C1 = A * B1
    for t in range(23):
        col = t
        for (i, j, k, c) in RANK23_3X3X3:
            if c == t + 1:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    # Next 23 columns: handle C2 = A * B2
    for t in range(23):
        col = 23 + t
        for (i, j, k, c) in RANK23_3X3X3:
            if c == t + 1:
                U[i * m + j][col] = 1
                V[j * p + (k + 3)][col] = 1  # B2 is columns 3-5, so k+3
                W[(k + 3) * n + i][col] = 1  # C2 is columns 3-5
    
    return {"U": U, "V": V, "W": W}
