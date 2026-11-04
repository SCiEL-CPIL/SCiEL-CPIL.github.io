def _decompose_4x5x5():
    """
    Decompose 4x5 * 5x5 -> 4x5.
    Strategy: Split n=4 into two 2x blocks.
    For each 2x5 * 5x5 -> 2x5 block:
    Split m=5 into 2+3, p=5 into 2+3.
    Sub-problems:
    1. 2x2 * 2x2 -> 2x2: Strassen rank 7
    2. 2x3 * 3x2 -> 2x2: Improved rank 11
    3. 2x2 * 2x3 -> 2x3: Improved rank 11
    4. 2x3 * 3x3 -> 2x3: Improved rank 17
    Total per block: 7+11+11+17 = 46
    Total: 46*2 = 92
    """
    n, m, p = 4, 5, 5
    
    total_rank = 92
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for block in range(2):
        i_offset = block * 2
        
        # Sub 1: 2x2 * 2x2 -> 2x2 (Strassen)
        U1, V1, W1, r1 = _strassen_2x2()
        for i in range(2):
            for j in range(2):
                for c in range(r1):
                    U[(i_offset + i) * m + j][col + c] = U1[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                for c in range(r1):
                    V[j * p + k][col + c] = V1[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                for c in range(r1):
                    W[k * n + (i_offset + i)][col + c] = W1[k * 2 + i][c]
        col += r1
        
        # Sub 2: 2x3 * 3x2 -> 2x2 (Improved, rank 11)
        U2, V2, W2, r2 = _decompose_2x3x2()
        for i in range(2):
            for j in range(3):
                for c in range(r2):
                    U[(i_offset + i) * m + (2 + j)][col + c] = U2[i * 3 + j][c]
        for j in range(3):
            for k in range(2):
                for c in range(r2):
                    V[(2 + j) * p + k][col + c] = V2[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                for c in range(r2):
                    W[k * n + (i_offset + i)][col + c] += W2[k * 2 + i][c]
        col += r2
        
        # Sub 3: 2x2 * 2x3 -> 2x3 (Improved, rank 11)
        U3, V3, W3, r3 = _decompose_2x2x3()
        for i in range(2):
            for j in range(2):
                for c in range(r3):
                    U[(i_offset + i) * m + j][col + c] = U3[i * 2 + j][c]
        for j in range(2):
            for k in range(3):
                for c in range(r3):
                    V[j * p + (2 + k)][col + c] = V3[j * 3 + k][c]
        for k in range(3):
            for i in range(2):
                for c in range(r3):
                    W[(2 + k) * n + (i_offset + i)][col + c] = W3[k * 2 + i][c]
        col += r3
        
        # Sub 4: 2x3 * 3x3 -> 2x3 (Improved, rank 17)
        U4, V4, W4, r4 = _decompose_2x3x3()
        for i in range(2):
            for j in range(3):
                for c in range(r4):
                    U[(i_offset + i) * m + (2 + j)][col + c] = U4[i * 3 + j][c]
        for j in range(3):
            for k in range(3):
                for c in range(r4):
                    V[(2 + j) * p + (2 + k)][col + c] = V4[j * 3 + k][c]
        for k in range(3):
            for i in range(2):
                for c in range(r4):
                    W[(2 + k) * n + (i_offset + i)][col + c] += W4[k * 2 + i][c]
        col += r4
    
    return U, V, W, col
