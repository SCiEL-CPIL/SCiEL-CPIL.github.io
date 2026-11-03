DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split n into 2 + 1
    n1, n2 = 2, 1
    
    # For the 1x5x6 part: tensor rank is 1 (all-ones tensor)
    # For the 2x5x6 part: use naive rank = 2*5*6 = 60
    rank1 = 1  # for 1x5x6
    rank2 = n1 * m * p  # for 2x5x6, naive
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # First, handle the 1x5x6 part (rows 2,3 of A, i.e., i=2,3 in 0-indexed, but n=3 so i=0,1,2)
    # Actually, n=3, so i ranges 0,1,2. Split: first 2 rows (i=0,1) use rank2, last 1 row (i=2) uses rank1
    # For the 1x5x6 part (i=2):
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
    col += rank1
    
    # For the 2x5x6 part (i=0,1): use naive decomposition
    for i in range(n1):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
