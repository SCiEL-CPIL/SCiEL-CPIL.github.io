<<<<<<< SEARCH
    # Let me use a rank-13 decomposition for 2x2x5
    rank = 13
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # n=2, m=2, p=5
    
    # Term 0: contributes to C[0,0] via A[0,0]*B[0,0]
    U[0][0] = 1  # A[0,0]
    V[0][0] = 1  # B[0,0]
    W[0][0] = 1  # C[0,0]
    
    # Term 1: contributes to C[0,1] via A[0,0]*B[0,1]
    U[0][1] = 1  # A[0,0]
    V[1][1] = 1  # B[0,1]
    W[1][1] = 1  # C[0,1]
    
    # Term 2: contributes to C[0,2] via A[0,0]*B[0,2]
    U[0][2] = 1  # A[0,0]
    V[2][2] = 1  # B[0,2]
    W[2][2] = 1  # C[0,2]
    
    # Term 3: contributes to C[0,3] via A[0,0]*B[0,3]
    U[0][3] = 1  # A[0,0]
    V[3][3] = 1  # B[0,3]
    W[3][3] = 1  # C[0,3]
    
    # Term 4: contributes to C[0,4] via A[0,0]*B[0,4]
    U[0][4] = 1  # A[0,0]
    V[4][4] = 1  # B[0,4]
    W[4][4] = 1  # C[0,4]
    
    # Term 5: contributes to C[0,0] via A[0,1]*B[1,0]
    U[1][5] = 1  # A[0,1]
    V[5][5] = 1  # B[1,0]
    W[0][5] = 1  # C[0,0]
    
    # Term 6: contributes to C[0,1] via A[0,1]*B[1,1]
    U[1][6] = 1  # A[0,1]
    V[6][6] = 1  # B[1,1]
    W[1][6] = 1  # C[0,1]
    
    # Term 7: contributes to C[0,2] via A[0,1]*B[1,2]
    U[1][7] = 1  # A[0,1]
    V[7][7] = 1  # B[1,2]
    W[2][7] = 1  # C[0,2]
    
    # Term 8: contributes to C[0,3] via A[0,1]*B[1,3]
    U[1][8] = 1  # A[0,1]
    V[8][8] = 1  # B[1,3]
    W[3][8] = 1  # C[0,3]
    
    # Term 9: contributes to C[0,4] via A[0,1]*B[1,4]
    U[1][9] = 1  # A[0,1]
    V[9][9] = 1  # B[1,4]
    W[4][9] = 1  # C[0,4]
    
    # Now for C[1,k], we need A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # Term 10: contributes to C[1,0] via A[1,0]*B[0,0]
    U[2][10] = 1  # A[1,0]
    V[0][10] = 1  # B[0,0]
    W[0][10] = 1  # C[1,0]
    
    # Term 11: contributes to C[1,1] via A[1,0]*B[0,1]
    U[2][11] = 1  # A[1,0]
    V[1][11] = 1  # B[0,1]
    W[1][11] = 1  # C[1,1]
    
    # Term 12: contributes to C[1,2] via A[1,0]*B[0,2]
    U[2][12] = 1  # A[1,0]
    V[2][12] = 1  # B[0,2]
    W[2][12] = 1  # C[1,2]
    
    # But we're missing C[1,3] and C[1,4]! This only gives 13 terms but doesn't cover all.
    # I need to reconsider.
    
    # Let me use a proper rank-13 construction.
    # The key is that we can use linear combinations to reduce the rank.
    
    # For now, let me use a correct but higher-rank decomposition.
    # I'll use rank 16 which is achievable.
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
=======
    # Use schoolbook decomposition with rank = n*m*p = 20
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
>>>>>>> REPLACE
