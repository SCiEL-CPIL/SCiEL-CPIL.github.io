def construct():
    n, m, p = 3, 4, 5
    # Split B into B1 (3x5) and B2 (1x5)
    # Split B1 into B11 (3x3) and B12 (3x2)
    # A is A1 (3x3) and A2 (3x1)
    
    # Term 1: A1 * B11 (3x3 by 3x3) -> Rank 9
    # Term 2: A1 * B12 (3x3 by 3x2) -> Rank 18
    # Term 3: A2 * B2 (3x1 by 1x5) -> Rank 15
    # Total Rank: 9 + 18 + 15 = 42
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Term 1: i in 0..2, j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Term 2: i in 0..2, j in 0..2, k in 3..4
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Term 3: i in 0..2, j in 3, k in 0..4
    for i in range(3):
        for j in [3]:
            for k in range(5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
