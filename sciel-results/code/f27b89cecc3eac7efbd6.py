DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Strassen 2x2 decomposition
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U2[0][0]=1; U2[3][0]=1
    V2[0][0]=1; V2[3][0]=1
    # M2 = (A10+A11)B00
    U2[2][1]=1; U2[3][1]=1
    V2[0][1]=1
    # M3 = A00(B10-B11)
    U2[0][2]=1
    V2[2][2]=1; V2[3][2]=-1
    # M4 = A11(B01-B00)
    U2[3][3]=1
    V2[1][3]=1; V2[0][3]=-1
    # M5 = (A00+A01)B11
    U2[0][4]=1; U2[1][4]=1
    V2[3][4]=1
    # M6 = (A10-A00)(B00+B01)
    U2[2][5]=1; U2[0][5]=-1
    V2[0][5]=1; V2[1][5]=1
    # M7 = (A01-A11)(B10+B11)
    U2[1][6]=1; U2[3][6]=-1
    V2[2][6]=1; V2[3][6]=1
    
    # W2: C = sum_r W2[:,r] * M_r
    # C00 = M1+M4-M5-M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M7
    W2[0][0]=1; W2[0][3]=1; W2[0][4]=-1; W2[0][6]=-1
    W2[1][2]=1; W2[1][4]=1
    W2[2][1]=1; W2[2][3]=1
    W2[3][0]=1; W2[3][1]=-1; W2[3][2]=1; W2[3][6]=1
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1*7 + r2
            for i in range(4):
                for j in range(4):
                    i1, i2 = i//2, i%2
                    j1, j2 = j//2, j%2
                    U[i*4+j][r] = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
            for j in range(4):
                for k in range(4):
                    j1, j2 = j//2, j%2
                    k1, k2 = k//2, k%2
                    V[j*4+k][r] = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
            for k in range(4):
                for i in range(4):
                    k1, k2 = k//2, k%2
                    i1, i2 = i//2, i%2
                    W[k*4+i][r] = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
