def construct():
    n = 10
    # MOLS of order 5
    # L1[i][j] = (i+j) % 5
    # L2[i][j] = (i+2*j) % 5
    
    # Construct M1 and M2 of order 10
    # M1[r][c] = 2 * L1[r%5][c%5] + (r%2)
    # M2[r][c] = 2 * L2[r%5][c%5] + (c%2)
    
    M1 = [[0]*n for _ in range(n)]
    M2 = [[0]*n for _ in range(n)]
    
    for r in range(n):
        for c in range(n):
            L1_val = (r % 5 + c % 5) % 5
            L2_val = (r % 5 + 2 * (c % 5)) % 5
            
            M1[r][c] = 2 * L1_val + (r % 2)
            M2[r][c] = 2 * L2_val + (c % 2)
            
    return [M1, M2]
