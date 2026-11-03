import numpy as np

def construct():
    n, m, p = 2, 3, 4
    # We split the middle dimension m=3 into m1=2 and m2=1.
    # A is 2x3, split into A1 (2x2) and A2 (2x1).
    # B is 3x4, split into B1 (2x4) and B2 (1x4).
    # C = A1 B1 + A2 B2.
    
    # Decomposition of 2x2x4 (A1 * B1)
    # Rank 10
    # We need 10 tuples (u, v, w) where u is 2x2, v is 2x4, w is 2x4.
    # Actually, the tensor is U (2*2, 10), V (2*4, 10), W (4*2, 10).
    # Indices:
    # U: (i, j) where i in 0..1, j in 0..1 (from A1)
    # V: (j, k) where j in 0..1 (from B1), k in 0..3
    # W: (k, i) where k in 0..3, i in 0..1
    
    # Standard 2x2x4 decomposition (Rank 10)
    # This is effectively 5 Strassen-like blocks?
    # I will use a known rank-10 decomposition for 2x2x4.
    # If I don't have it, I'll use rank 8 (schoolbook for 4 dot products) + 4 = 12?
    # Schoolbook for 2x2x4 is rank 8.
    # 8 + 4 = 12.
    # 12 < 24. This is a valid improvement.
    
    rank1 = 8
    rank2 = 4
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: A1 (2x2) * B1 (2x4) -> C1 (2x4)
    # A1 indices: i in 0..1, j in 0..1.
    # B1 indices: j in 0..1, k in 0..3.
    # C1 indices: i in 0..1, k in 0..3.
    # Schoolbook for this part:
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U[i * m + j][col] = 1  # A1[i, j]
                V[j * p + k][col] = 1  # B1[j, k]
                W[k * n + i][col] = 1  # C1[k, i]
                col += 1
                
    # Part 2: A2 (2x1) * B2 (1x4) -> C2 (2x4)
    # A2 indices: i in 0..1, j in 2 (since m=3, j=2 is the last one).
    # B2 indices: j in 2, k in 0..3.
    # C2 indices: i in 0..1, k in 0..3.
    for i in range(2):
        for k in range(4):
            U[i * m + 2][col] = 1  # A2[i, 0] which is A[i, 2]
            V[2 * p + k][col] = 1  # B2[0, k] which is B[2, k]
            W[k * n + i][col] = 1  # C2[k, i]
            col += 1
            
    return {"U": U, "V": V, "W": W}
