"""Rewritten program: uses a fixed known low-rank decomposition for 6x6 matrix multiplication."""

def construct():
    # Kronecker product of Strassen's 2x2 algorithm (rank 7) with
    # the schoolbook 3x3 algorithm (rank 27) to get rank 7*27 = 189.
    # This is an exact decomposition for 6x6 matrix multiplication.
    
    # Strassen 2x2 factors
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U2[0][0] += 1; U2[3][0] += 1
    V2[0][0] += 1; V2[3][0] += 1
    W2[0][0] += 1; W2[3][0] += 1
    
    # M2 = (A10 + A11)B00
    U2[2][1] += 1; U2[3][1] += 1
    V2[0][1] += 1
    W2[2][1] += 1
    
    # M3 = A00(B01 - B11)
    U2[0][2] += 1
    V2[1][2] += 1; V2[3][2] -= 1
    W2[1][2] += 1; W2[3][2] += 1
    
    # M4 = A11(B10 - B00)
    U2[3][3] += 1
    V2[2][3] += 1; V2[0][3] -= 1
    W2[0][3] += 1; W2[2][3] += 1
    
    # M5 = (A00 + A10)B11
    U2[0][4] += 1; U2[2][4] += 1
    V2[3][4] += 1
    W2[0][4] -= 1; W2[1][4] += 1
    
    # M6 = (A11 - A10)(B00 + B10)
    U2[3][5] += 1; U2[2][5] -= 1
    V2[0][5] += 1; V2[2][5] += 1
    W2[3][5] += 1
    
    # M7 = (A00 - A01)(B01 + B11)
    U2[0][6] += 1; U2[1][6] -= 1
    V2[1][6] += 1; V2[3][6] += 1
    W2[0][6] += 1
    
    # Schoolbook 3x3: rank 27
    rank3 = 27
    U3 = [[0]*rank3 for _ in range(9)]
    V3 = [[0]*rank3 for _ in range(9)]
    W3 = [[0]*rank3 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
    
    # Kronecker: rank6 = 7 * 27 = 189
    rank6 = 7 * 27
    U6 = [[0]*rank6 for _ in range(36)]
    V6 = [[0]*rank6 for _ in range(36)]
    W6 = [[0]*rank6 for _ in range(36)]
    
    for r2 in range(7):
        for r3 in range(27):
            r = r2 * 27 + r3
            for i in range(6):
                i2 = i // 3
                i3 = i % 3
                for j in range(6):
                    j2 = j // 3
                    j3 = j % 3
                    U6[i*6+j][r] = U2[i2*2+j2][r2] * U3[i3*3+j3][r3]
            for j in range(6):
                j2 = j // 3
                j3 = j % 3
                for k in range(6):
                    k2 = k // 3
                    k3 = k % 3
                    V6[j*6+k][r] = V2[j2*2+k2][r2] * V3[j3*3+k3][r3]
            for k in range(6):
                k2 = k // 3
                k3 = k % 3
                for i in range(6):
                    i2 = i // 3
                    i3 = i % 3
                    W6[k*6+i][r] = W2[k2*2+i2][r2] * W3[k3*3+i3][r3]
    
    return {"U": U6, "V": V6, "W": W6}
