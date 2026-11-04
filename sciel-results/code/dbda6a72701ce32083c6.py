<<<<<<< SEARCH
    # Build the decomposition
    if _time_remaining() > 5.0:
        # PART 1: 4x4 * 4x4 (columns 0-3 of B) -> rank 49
        strassen_4x4(0, 0, 0)
    
    if _time_remaining() > 3.0:
        # PART 2: 4x4 * 4x2 (columns 4-5 of B) -> rank 28
        # For each pair of rows in C (rb = 0, 1):
        #   C[rb*2:rb*2+2, 4:6] = A[rb*2:rb*2+2, 0:2]*B[0:2, 4:6] + A[rb*2:rb*2+2, 2:4]*B[2:4, 4:6]
        for rb in range(2):
            if _time_remaining() < 2.0:
                break
            strassen_2x2(rb*2, 0, 4)
            strassen_2x2(rb*2, 2, 4)
    
    rank = len(U_cols)
    
    # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Verify the decomposition
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: schoolbook rank-96
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
=======
    # Build the decomposition
    if _time_remaining() > 5.0:
        # PART 1: 4x4 * 4x4 (columns 0-3 of B) -> rank 49
        strassen_4x4(0, 0, 0)
    
    if _time_remaining() > 3.0:
        # PART 2: 4x4 * 4x2 (columns 4-5 of B) -> rank 28
        # For each pair of rows in C (rb = 0, 1):
        #   C[rb*2:rb*2+2, 4:6] = A[rb*2:rb*2+2, 0:2]*B[0:2, 4:6] + A[rb*2:rb*2+2, 2:4]*B[2:4, 4:6]
        for rb in range(2):
            if _time_remaining() < 2.0:
                break
            strassen_2x2(rb*2, 0, 4)
            strassen_2x2(rb*2, 2, 4)
    
    rank = len(U_cols)
    
    # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Verify the decomposition
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Fallback: schoolbook rank-96
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
>>>>>>> REPLACE

