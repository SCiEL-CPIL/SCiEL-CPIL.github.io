import numpy as np

def construct():
    # Strassen 2x2x2 decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]

    # Part 1: 4x4x4 using Strassen
    # Rank 49
    n, m, p = 4, 4, 4
    rank1 = 49
    U1 = [[0] * rank1 for _ in range(n * m)]
    V1 = [[0] * rank1 for _ in range(m * p)]
    W1 = [[0] * rank1 for _ in range(p * n)]
    
    for s in range(7):
        for t in range(7):
            c = s * 7 + t
            for i1 in range(2):
                for i0 in range(2):
                    i = i1 * 2 + i0
                    U1[i][c] = U2[i1][s] * U2[i0][t]
            
            for j1 in range(2):
                for j0 in range(2):
                    j = j1 * 2 + j0
                    V1[j][c] = V2[j1][s] * V2[j0][t]
            
            for k1 in range(2):
                for k0 in range(2):
                    k = k1 * 2 + k0
                    W1[k][c] = W2[k1][s] * W2[k0][t]

    # Part 2: 4x4x1 using Schoolbook
    # Rank 16
    n2, m2, p2 = 4, 4, 1
    rank2 = 16
    U2_part = [[0] * rank2 for _ in range(n2 * m2)]
    V2_part = [[0] * rank2 for _ in range(m2 * p2)]
    W2_part = [[0] * rank2 for _ in range(p2 * n2)]
    
    col = 0
    for i in range(n2):
        for j in range(m2):
            U2_part[i * m2 + j][col] = 1
            V2_part[j * p2 + 0][col] = 1 # B index is just j
            W2_part[0 * n2 + i][col] = 1 # C index is just i
            col += 1

    # Combine
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(20)] # 4x5 = 20
    W = [[0] * total_rank for _ in range(20)] # 4x5 = 20 (p*n = 4*5? No, p=5, n=4. W is p*n x rank. 20x65)
    
    # Wait, W shape is (p*n, rank). p=5, n=4. 20x65.
    # V shape is (m*p, rank). m=4, p=5. 20x65.
    # U shape is (n*m, rank). n=4, m=4. 16x65.
    
    # Copy Part 1 (4x4x4)
    # U1 is 16x49. U is 16x65.
    for i in range(16):
        for c in range(49):
            U[i][c] = U1[i][c]
            
    # V1 is 16x49 (4x4). V is 20x65 (4x5).
    # V1 corresponds to B columns 0..3.
    for j in range(16):
        for c in range(49):
            V[j][c] = V1[j][c]
            
    # W1 is 16x49 (4x4). W is 20x65 (5x4).
    # W1 corresponds to C columns 0..3.
    # W index is k*n + i. k is 0..3 for first 4 columns.
    for k in range(16):
        for c in range(49):
            W[k][c] = W1[k][c]

    # Copy Part 2 (4x4x1)
    # U2_part is 16x16.
    for i in range(16):
        for c in range(16):
            U[i][49 + c] = U2_part[i][c]
            
    # V2_part is 4x16 (4x1).
    # V index for column 4 is j*5 + 4.
    for j in range(4):
        for c in range(16):
            V[j * 5 + 4][49 + c] = V2_part[j][c]
            
    # W2_part is 4x16 (1x4).
    # W index for column 4 is 4*4 + i = 16 + i.
    for i in range(4):
        for c in range(16):
            W[16 + i][49 + c] = W2_part[i][c]

    return {"U": U, "V": V, "W": W}
