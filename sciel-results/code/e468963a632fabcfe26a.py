import time
import numpy as np
from collections import defaultdict

# Known rank-19 decomposition for 3x3x4 matrix multiplication.
# U indices: 0-8 (row,col in A), V indices: 0-11 (col in A, row in B), W indices: 0-11 (row in B, col in A)
# This is a known minimal decomposition from the literature.
# Format: (U_index, V_index, W_index)
RANK19_DECOMP = [
    (0, 0, 0), (0, 4, 3), (0, 8, 6), (1, 3, 1), (1, 7, 4), (1, 11, 7),
    (2, 2, 2), (2, 6, 5), (2, 10, 8), (3, 1, 1), (3, 5, 4), (3, 9, 7),
    (4, 0, 3), (4, 4, 6), (4, 8, 9), (5, 3, 2), (5, 7, 5), (5, 11, 8),
    (6, 2, 0), (6, 6, 1), (6, 10, 2), (7, 1, 4), (7, 5, 7), (7, 9, 10),
    (8, 0, 9), (8, 4, 12), (8, 8, 15), (9, 3, 3), (9, 7, 6), (9, 11, 9),
    (10, 2, 0), (10, 6, 1), (10, 10, 2), (11, 1, 5), (11, 5, 8), (11, 9, 11),
    (12, 0, 6), (12, 4, 9), (12, 8, 12), (13, 3, 2), (13, 7, 5), (13, 11, 8),
    (14, 2, 0), (14, 6, 1), (14, 10, 2), (15, 1, 4), (15, 5, 7), (15, 9, 10),
    (16, 0, 9), (16, 4, 12), (16, 8, 15), (17, 3, 3), (17, 7, 6), (17, 11, 9),
    (18, 2, 0), (18, 6, 1), (18, 10, 2)
]
# Wait, the above list is wrong. I need a valid 3x3x4 rank-19 decomposition.
# Let me use a simpler approach: try to find a valid one or use a known correct one.
# Actually, let's just use the schoolbook 72 rank but try to improve with a search.
# Or better, let me use a known valid 3x3x4 decomposition.
# A valid 3x3x4 rank-19 decomposition exists. Let me construct one.
# Actually, the safest bet is to use the schoolbook 72 and not risk an invalid decomposition.

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 3, 8
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
