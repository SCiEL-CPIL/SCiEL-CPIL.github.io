import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 decomposition
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
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, 0, 0, 0, 0, 1, 1]   # C11
    ]

    # Block structure for 4x4 x 4x7:
    # A is 4x4, viewed as 2x2 blocks of 2x2
    # B is 4x7, viewed as 2x3 blocks: [2x2, 2x2, 2x1]
    # C is 4x7, viewed as 2x3 blocks: [2x2, 2x2, 2x1]
    
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B02 (rows 0-1, cols 4-6), 
    #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3), B12 (rows 2-3, cols 4-6)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C02 (rows 0-1, cols 4-6),
    #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3), C12 (rows 2-3, cols 4-6)
    
    # We need to compute C = A * B where:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Using Strassen for the 2x2 block multiplication:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For the 2x1 blocks (column 2 of C), we need similar decompositions
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # We can use additional Strassen-like terms for these
    # M8 = (A00+A11)B02
    # M9 = (A10+A11)B12
    # M10 = A00(B02)
    # M11 = A11(B12)
    
    # Actually, let's use a simpler approach: treat each column block separately
    # For columns 0-1: use Strassen on 2x2 blocks
    # For columns 2-3: use Strassen on 2x2 blocks  
    # For columns 4-6: use schoolbook or another decomposition
    
    # Let me reconsider. The 4x7 can be seen as two 4x2 blocks and one 4x3 block,
    # or as seven 4x1 columns. Let's use the block structure with 2x2 and 2x1.
    
    # I'll implement this by generating the rank-98 decomposition directly.
    # The 98 rank comes from: 7 (Strassen) * 2 (for the two 2x2 column blocks) + 14 (for the 2x1 column block) = 28? No, that's not right.
    
    # Actually, 98 = 7 * 14? No. Let me think differently.
    # 4x4x4 has rank 49 (Strassen^2). For 4x4x7, if we can do it with 98, that's 49 * 2.
    # Maybe we can view 4x7 as two 4x3.5? No.
    
    # Let me just implement the schoolbook for now to ensure correctness, then optimize.
    # Schoolbook rank: 4*4*7 = 112
    
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
