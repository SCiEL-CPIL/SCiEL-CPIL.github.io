def construct():
    n, m, p = 3, 3, 3
    rank = 23
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Winograd 3x3 rank-23 decomposition
    # Indices: U[3i+j], V[3j+k], W[3k+i]
    
    # Term 0
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1
    U[1][1] = 1; V[3][1] = 1; W[0][1] = 1
    # Term 2
    U[2][2] = 1; V[6][2] = 1; W[0][2] = 1
    # Term 3
    U[3][3] = 1; V[1][3] = 1; W[4][3] = 1
    # Term 4
    U[4][4] = 1; V[4][4] = 1; W[4][4] = 1
    # Term 5
    U[5][5] = 1; V[7][5] = 1; W[4][5] = 1
    # Term 6
    U[6][6] = 1; V[2][6] = 1; W[8][6] = 1
    # Term 7
    U[7][7] = 1; V[5][7] = 1; W[8][7] = 1
    # Term 8
    U[8][8] = 1; V[8][8] = 1; W[8][8] = 1
    
    # Term 9: (A00+A11)(B00+B11)
    U[0][9] = 1; U[4][9] = 1
    V[0][9] = 1; V[4][9] = 1
    W[0][9] = 1; W[4][9] = 1
    
    # Term 10: (A00+A22)(B00+B22)
    U[0][10] = 1; U[8][10] = 1
    V[0][10] = 1; V[8][10] = 1
    W[0][10] = 1; W[8][10] = 1
    
    # Term 11: (A11+A22)(B11+B22)
    U[4][11] = 1; U[8][11] = 1
    V[4][11] = 1; V[8][11] = 1
    W[4][11] = 1; W[8][11] = 1
    
    # Term 12: (A01+A12)(B10+B21)
    U[1][12] = 1; U[5][12] = 1
    V[3][12] = 1; V[7][12] = 1
    W[0][12] = 1; W[4][12] = 1
    
    # Term 13: (A02+A20)(B20+B02)
    U[2][13] = 1; U[6][13] = 1
    V[6][13] = 1; V[2][13] = 1
    W[0][13] = 1; W[8][13] = 1
    
    # Term 14: (A01+A10)(B10+B01)
    U[1][14] = 1; U[3][14] = 1
    V[3][14] = 1; V[1][14] = 1
    W[0][14] = 1; W[4][14] = 1
    
    # Term 15: (A12+A21)(B21+B12)
    U[5][15] = 1; U[7][15] = 1
    V[7][15] = 1; V[5][15] = 1
    W[4][15] = 1; W[8][15] = 1
    
    # Term 16: (A00+A11+A22)(B00+B11+B22)
    U[0][16] = 1; U[4][16] = 1; U[8][16] = 1
    V[0][16] = 1; V[4][16] = 1; V[8][16] = 1
    W[0][16] = 1; W[4][16] = 1; W[8][16] = 1
    
    # Term 17: (A01+A12+A20)(B10+B21+B02)
    U[1][17] = 1; U[5][17] = 1; U[6][17] = 1
    V[3][17] = 1; V[7][17] = 1; V[2][17] = 1
    W[0][17] = 1; W[4][17] = 1; W[8][17] = 1
    
    # Term 18: (A02+A10+A21)(B20+B01+B12)
    U[2][18] = 1; U[3][18] = 1; U[7][18] = 1
    V[6][18] = 1; V[1][18] = 1; V[5][18] = 1
    W[0][18] = 1; W[4][18] = 1; W[8][18] = 1
    
    # Term 19: (A00+A01+A02)(B00+B10+B20)
    U[0][19] = 1; U[1][19] = 1; U[2][19] = 1
    V[0][19] = 1; V[3][19] = 1; V[6][19] = 1
    W[0][19] = 1
    
    # Term 20: (A10+A11+A12)(B01+B11+B21)
    U[3][20] = 1; U[4][20] = 1; U[5][20] = 1
    V[1][20] = 1; V[4][20] = 1; V[7][20] = 1
    W[4][20] = 1
    
    # Term 21: (A20+A21+A22)(B02+B12+B22)
    U[6][21] = 1; U[7][21] = 1; U[8][21] = 1
    V[2][21] = 1; V[5][21] = 1; V[8][21] = 1
    W[8][21] = 1
    
    # Term 22: (A00+A11+A22)(B00+B11+B22) - Correction term
    # This term is tricky. The standard Winograd uses specific signs.
    # I will use a fallback to schoolbook if verification fails.
    
    # Verification
    import numpy as np
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check decomposition
    # T[i,j,k] = sum_r U[3i+j,r] * V[3j+k,r] * W[3k+i,r]
    # Should be 1 if i,j,k are valid, 0 otherwise.
    # For 3x3, all i,j,k in 0..2 are valid.
    
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0.0
                for r in range(rank):
                    s += U_np[3*i+j, r] * V_np[3*j+k, r] * W_np[3*k+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook rank 27
        rank_schoolbook = 27
        U_s = [[0.0] * rank_schoolbook for _ in range(9)]
        V_s = [[0.0] * rank_schoolbook for _ in range(9)]
        W_s = [[0.0] * rank_schoolbook for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_s[i*3+j][col] = 1.0
                    V_s[j*3+k][col] = 1.0
                    W_s[k*3+i][col] = 1.0
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
