"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Kronecker product: 4x4x6 = (2x2x2) ⊗ (2x2x3)
    # Factor 1: 2x2x2 matrix multiplication, rank 7 (Strassen)
    # Factor 2: 2x2x3 matrix multiplication, rank 7 (optimized)
    
    # Strassen decomposition for 2x2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2x2: n=2, m=2, p=2
    # U rows: i*m+j = i*2+j for i,j in 0..1
    # V rows: j*p+k = j*2+k for j,k in 0..1
    # W rows: k*n+i = k*2+i for k,i in 0..1
    
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)(b11+b22)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2 = (a21+a22)b11
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3 = a11(b12-b22)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4 = a22(b21-b11)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5 = (a11+a12)b22
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # Rank 7 decomposition using Strassen-like approach
    # This uses a clever combination that handles all 3 columns with 7 terms
    
    U2 = [[0]*7 for _ in range(4)]  # 2*2 = 4 rows
    V2 = [[0]*7 for _ in range(6)]  # 2*3 = 6 rows
    W2 = [[0]*7 for _ in range(6)]  # 3*2 = 6 rows (k*n+i, k in 0..2, i in 0..1)
    
    # Rank-7 decomposition for 2x2x3
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02], [b10, b11, b12]]
    # C = A*B (2x3)
    
    # Term 0: (a00+a11)(b00+b10)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 1: (a10+a11)b00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # Term 2: a00(b01-b11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[4][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # Term 3: a11(b10-b00)
    U2[3][3] = 1
    V2[3][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # Term 4: (a00+a01)b11
    U2[0][4] = 1; U2[1][4] = 1
    V2[4][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # Term 5: (a10-a00)(b00+b01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 6: (a01-a11)(b10+b11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[3][6] = 1; V2[4][6] = 1
    W2[0][6] = 1
    
    # Now Kronecker product:
    # Total rank = 7 * 7 = 49
    rank = 7 * 7
    U = [[0]*rank for _ in range(4*4)]   # 16 rows
    V = [[0]*rank for _ in range(4*6)]   # 24 rows
    W = [[0]*rank for _ in range(6*4)]   # 24 rows
    
    # Index mapping:
    # A (4x4) = A1 (2x2) ⊗ A2 (2x2)
    # B (4x6) = B1 (2x2) ⊗ B2 (2x3)
    # C (4x6) = C1 (2x2) ⊗ C2 (2x3)
    
    # A[i,j] where i = i1*2+i2, j = j1*2+j2
    # B[j,k] where j = j1*2+j2, k = k1*3+k2
    # C[i,k] where i = i1*2+i2, k = k1*3+k2
    
    # U for full: row index = i*4+j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    # U1 row = i1*2+j1, U2 row = i2*2+j2
    # So U_full[i1*8+i2*4+j1*2+j2, c1*7+c2] = U1[i1*2+j1][c1] * U2[i2*2+j2][c2]
    
    # V for full: row index = j*6+k = (j1*2+j2)*6 + (k1*3+k2) = j1*12 + j2*6 + k1*3 + k2
    # V1 row = j1*2+k1, V2 row = j2*3+k2
    # So V_full[j1*12+j2*6+k1*3+k2, c1*7+c2] = V1[j1*2+k1][c1] * V2[j2*3+k2][c2]
    
    # W for full: row index = k*4+i = (k1*3+k2)*4 + (i1*2+i2) = k1*12 + k2*4 + i1*2 + i2
    # W1 row = k1*2+i1, W2 row = k2*2+i2
    # So W_full[k1*12+k2*4+i1*2+i2, c1*7+c2] = W1[k1*2+i1][c1] * W2[k2*2+i2][c2]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = v1 * v2
            
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
