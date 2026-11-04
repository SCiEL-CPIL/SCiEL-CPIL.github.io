"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 3x3 schoolbook decomposition
    n3 = 3
    rank3 = n3 * n3 * n3
    U3 = [[0] * rank3 for _ in range(n3 * n3)]
    V3 = [[0] * rank3 for _ in range(n3 * n3)]
    W3 = [[0] * rank3 for _ in range(n3 * n3)]
    r3 = 0
    for i in range(n3):
        for j in range(n3):
            for k in range(n3):
                U3[i * n3 + j][r3] = 1
                V3[j * n3 + k][r3] = 1
                W3[k * n3 + i][r3] = 1
                r3 += 1

    # 2x2 Strassen decomposition
    n2 = 2
    rank2 = 7
    U2 = [[0] * rank2 for _ in range(n2 * n2)]
    V2 = [[0] * rank2 for _ in range(n2 * n2)]
    W2 = [[0] * rank2 for _ in range(n2 * n2)]
    
    # M0: (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M1: (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # M2: A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # M3: A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1; W2[0][3] = -1
    
    # M4: (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M5: (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M6: (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Combine using Kronecker product
    rank = rank3 * rank2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r3_idx in range(rank3):
        for r2_idx in range(rank2):
            r = r3_idx * rank2 + r2_idx
            
            for i in range(n3):
                for j in range(n3):
                    for i2 in range(n2):
                        for j2 in range(n2):
                            # U index: (i * n2 + i2) * m + (j * n2 + j2)
                            # m = 6 = n3 * n2
                            u_row = (i * n2 + i2) * m + (j * n2 + j2)
                            val = U3[i * n3 + j][r3_idx] * U2[i2 * n2 + j2][r2_idx]
                            if val != 0:
                                U[u_row][r] = val
                                
            for j in range(n3):
                for k in range(n3):
                    for j2 in range(n2):
                        for k2 in range(n2):
                            v_row = (j * n2 + j2) * p + (k * n2 + k2)
                            val = V3[j * n3 + k][r3_idx] * V2[j2 * n2 + k2][r2_idx]
                            if val != 0:
                                V[v_row][r] = val
                                
            for k in range(n3):
                for i in range(n3):
                    for k2 in range(n2):
                        for i2 in range(n2):
                            w_row = (k * n2 + k2) * n + (i * n2 + i2)
                            val = W3[k * n3 + i][r3_idx] * W2[k2 * n2 + i2][r2_idx]
                            if val != 0:
                                W[w_row][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
