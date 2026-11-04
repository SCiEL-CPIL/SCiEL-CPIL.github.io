import numpy as np

def construct():
    n, m, p = 4, 4, 6
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 steps
    # (alpha, beta, gamma)
    # alpha, beta, gamma are dicts {(i,j): coeff}
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    # Part 1: 4x4x4 multiplication (cols 0-3 of B, cols 0-3 of C)
    # Rank 49
    for s1 in range(7):
        a1, b1, c1 = strassen_steps[s1]
        for s2 in range(7):
            a2, b2, c2 = strassen_steps[s2]
            col = s1 * 7 + s2
            
            # U: A indices (row, col) -> index row*m + col
            for (bi, bj), va1 in a1.items():
                for (si, sj), va2 in a2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += va1 * va2
                    
            # V: B indices (row, col) -> index row*p + col
            # B is 4x4 here.
            for (bi, bj), vb1 in b1.items():
                for (si, sj), vb2 in b2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += vb1 * vb2
                    
            # W: C indices (row, col) -> index col*n + row
            # C is 4x4 here.
            for (bi, bj), vc1 in c1.items():
                for (si, sj), vc2 in c2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += vc1 * vc2

    # Part 2: 4x4x2 multiplication (cols 4-5 of B, cols 4-5 of C)
    # Rank 28
    # A is 4x4. B is 4x2. C is 4x2.
    # Partition A into 2x2 blocks: A00, A01, A10, A11
    # Partition B into 2x2 blocks vertically: B0 (rows 0-1), B1 (rows 2-3)
    # Partition C into 2x2 blocks vertically: C0 (rows 0-1), C1 (rows 2-3)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # We perform 4 separate 2x2x2 multiplications.
    # 1. A00 * B0 -> contributes to C0
    # 2. A01 * B1 -> contributes to C0
    # 3. A10 * B0 -> contributes to C1
    # 4. A11 * B1 -> contributes to C1
    
    mults = [
        # (A_block_row, A_block_col, B_block_row, C_block_row)
        (0, 0, 0, 0), # A00, B0 -> C0
        (0, 1, 1, 0), # A01, B1 -> C0
        (1, 0, 0, 1), # A10, B0 -> C1
        (1, 1, 1, 1)  # A11, B1 -> C1
    ]
    
    for m_idx, (ar, ac, br, cr) in enumerate(mults):
        for s2 in range(7):
            a2, b2, c2 = strassen_steps[s2]
            col = 49 + m_idx * 7 + s2
            
            # U: A block (ar, ac)
            for (si, sj), va2 in a2.items():
                r = (ar * 2 + si) * m + (ac * 2 + sj)
                U[r][col] += va2
                
            # V: B block (br, 0) -> global rows br*2+si, cols 4+sj
            for (si, sj), vb2 in b2.items():
                r = (br * 2 + si) * p + (4 + sj)
                V[r][col] += vb2
                
            # W: C block (cr, 0) -> global rows cr*2+si, cols 4+sj
            # W index: col*n + row
            for (si, sj), vc2 in c2.items():
                r = (4 + sj) * n + (cr * 2 + si)
                W[r][col] += vc2

    return {"U": U, "V": V, "W": W}
